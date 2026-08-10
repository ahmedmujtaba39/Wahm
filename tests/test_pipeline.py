import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import finalize
import generate
import analyze_results
import evaluate_layer3
import prepare_layer3
import score_layer1
import translate
from judge_data import (grouped_split, grouped_train_validation_test_split,
                        load_judge_rows)
from wahm_text import normalize_arabic, token_f1
from scripts.verify_hf_snapshot import run as verify_hf_snapshot


ROOT = Path(__file__).resolve().parents[1]


class ArabicTextTests(unittest.TestCase):
    def test_normalization_unifies_common_forms(self):
        self.assertEqual(normalize_arabic("إجابةٌ"), "اجابه")

    def test_token_f1_is_symmetric(self):
        left = token_f1("أين يقع المسجد؟", "وين يقع المسجد")
        right = token_f1("وين يقع المسجد", "أين يقع المسجد؟")
        self.assertEqual(left, right)


class LayerOneTests(unittest.TestCase):
    def test_clean_exact_reference(self):
        coverage, decision, reasons = score_layer1.score_answer(
            "فازت الأوروغواي ببطولتين.", "بطولتين")
        self.assertEqual((coverage, decision, reasons), (1.0, "clean", []))

    def test_partial_reference_defers(self):
        _, decision, _ = score_layer1.score_answer("بيير كوري", "بيير كوري وأخوه جاك")
        self.assertEqual(decision, "defer")

    def test_no_reference_overlap_is_hallucinated(self):
        _, decision, _ = score_layer1.score_answer("لا أعرف", "بطولتين")
        self.assertEqual(decision, "hallucinated")

    def test_degeneration_overrides_clean_coverage(self):
        _, decision, reasons = score_layer1.score_answer(
            "بطولتين 中文", "بطولتين")
        self.assertEqual(decision, "hallucinated")
        self.assertIn("foreign_script", reasons)

    def test_decimal_comma_is_not_treated_as_variant_separator(self):
        variants = score_layer1.gold_variants(
            "37,5 مليون, 37,5 مليون نسمة تقريبا")
        self.assertEqual(variants[0], "37,5 مليون")
        self.assertEqual(len(variants), 2)

    def test_retry_log_keeps_latest_success_only(self):
        base = {"qid": "q1", "variety": "msa", "condition": "direct",
                "model": "m"}
        rows = [dict(base, answer="", error="timeout"),
                dict(base, answer="answer", error="")]
        canonical = score_layer1.canonical_generations(rows)
        self.assertEqual(len(canonical), 1)
        self.assertEqual(canonical[0]["answer"], "answer")


class TranslationTests(unittest.TestCase):
    def test_existing_confirmed_gulf_file_loads(self):
        old = os.getcwd()
        os.chdir(ROOT)
        try:
            pairs, _ = translate.load_exemplars("gulf")
        finally:
            os.chdir(old)
        self.assertEqual(len(pairs), 10)

    def test_valid_exemplar_file_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exemplars_test.csv"
            with path.open("w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=[
                    "msa_question", "dialect_question", "sub_variety"])
                writer.writeheader()
                for i in range(5):
                    writer.writerow({"msa_question": f"msa {i}",
                                     "dialect_question": f"dia {i}",
                                     "sub_variety": "Test variety"})
            old = os.getcwd()
            os.chdir(directory)
            try:
                pairs, variety = translate.load_exemplars("test")
            finally:
                os.chdir(old)
            self.assertEqual(len(pairs), 5)
            self.assertEqual(variety, "Test variety")

    def test_translation_resumes_without_repeating_paid_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "wahm_seed_msa.csv").write_text(
                "qid,question_msa,gold_answer\nq1,سؤال,جواب\n", encoding="utf-8")
            with (directory / "exemplars_test.csv").open(
                    "w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=[
                    "msa_question", "dialect_question", "sub_variety"])
                writer.writeheader()
                for i in range(5):
                    writer.writerow({"msa_question": f"msa {i}",
                                     "dialect_question": f"dia {i}",
                                     "sub_variety": "Test variety"})

            calls = []
            original_seed, original_call = translate.SEED, translate.call
            translate.SEED = "wahm_seed_msa.csv"
            translate.DIALECT_NAMES["test"] = "Test Arabic"
            translate.call = lambda *args, **kwargs: (
                calls.append(args) or "نص", "resolved-model")
            old = os.getcwd()
            os.chdir(directory)
            try:
                translate.translate("test", model="requested-model")
                translate.translate("test", model="requested-model")
            finally:
                os.chdir(old)
                translate.SEED, translate.call = original_seed, original_call
                del translate.DIALECT_NAMES["test"]
            self.assertEqual(len(calls), 2)  # translation + backtranslation once


class GenerationTests(unittest.TestCase):
    def test_msa_pivot_records_both_resolved_models(self):
        class Completions:
            calls = 0

            def create(self, **kwargs):
                from types import SimpleNamespace
                self.calls += 1
                text = "سؤال فصيح" if self.calls == 1 else "إجابة"
                return SimpleNamespace(
                    model=f"resolved-{self.calls}",
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content=text))])

        from types import SimpleNamespace
        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        generate._clients["gpt4o"] = client
        try:
            answer, error, pivot, resolved, pivot_model = generate.generate_one(
                "gpt4o", "سؤال خليجي", "gulf", "msa_pivot", 0.0)
        finally:
            generate._clients.clear()
        self.assertEqual(error, "")
        self.assertEqual((pivot, answer), ("سؤال فصيح", "إجابة"))
        self.assertEqual((pivot_model, resolved), ("resolved-1", "resolved-2"))


class FinalizationTests(unittest.TestCase):
    def test_rewrite_survives_low_backtranslation_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            candidate = directory / "candidates_test.csv"
            validation = directory / "validation_test.csv"
            candidate.write_text(
                "qid,question_msa,gold_answer,dialect_candidate,backtranslation_msa,sub_variety\n"
                "q1,السؤال,الجواب,ترجمة سيئة,مختلف تماما,Test\n",
                encoding="utf-8")
            validation.write_text(
                "qid,APPROVE_or_REWRITE,your_rewrite,fluency_1_5\n"
                "q1,REWRITE,ترجمة صحيحة,5\n", encoding="utf-8")
            old = os.getcwd()
            os.chdir(directory)
            try:
                metrics = finalize.run("test", min_backtrans=0.9)
            finally:
                os.chdir(old)
            self.assertEqual(metrics["n_final_benchmark"], 1)
            self.assertEqual(metrics["rewrite_rate"], 1.0)


class JudgeSplitTests(unittest.TestCase):
    def test_question_groups_do_not_leak(self):
        rows = load_judge_rows(ROOT / "Judge_train.csv.csv")
        train, test = grouped_split(rows)
        train_groups = {rows[i]["sample_index"] for i in train}
        test_groups = {rows[i]["sample_index"] for i in test}
        self.assertTrue(train_groups.isdisjoint(test_groups))

    def test_train_validation_test_groups_are_disjoint(self):
        rows = load_judge_rows(ROOT / "Judge_train.csv.csv")
        partitions = grouped_train_validation_test_split(rows)
        groups = [{rows[i]["sample_index"] for i in partition}
                  for partition in partitions]
        self.assertTrue(groups[0].isdisjoint(groups[1]))
        self.assertTrue(groups[0].isdisjoint(groups[2]))
        self.assertTrue(groups[1].isdisjoint(groups[2]))

    def test_snapshot_audit_detects_modified_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            model_file = snapshot / "model.bin"
            model_file.write_bytes(b"weights")
            audit = {
                "model_id": "model",
                "revision": "commit",
                "snapshot_dir": str(snapshot),
                "files": [{
                    "path": "model.bin",
                    "bytes": len(b"weights"),
                    "sha256": hashlib.sha256(b"weights").hexdigest(),
                }],
            }
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            verify_hf_snapshot(audit_path, "model", "commit")
            model_file.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                verify_hf_snapshot(audit_path, "model", "commit")


class AnalysisTests(unittest.TestCase):
    def test_hds_uses_only_qids_paired_with_msa(self):
        def row(qid, variety, decision):
            return {"qid": qid, "variety": variety, "condition": "direct",
                    "model": "m", "family": "test",
                    "combined_decision": decision, "degeneration_reasons": ""}

        rows = [row("q1", "msa", "clean"),
                row("q2", "msa", "hallucinated"),
                row("q1", "gulf", "hallucinated"),
                row("q3", "gulf", "hallucinated")]
        result = analyze_results.analyze(rows)[0]
        self.assertEqual(result["n_paired"], 1)
        self.assertEqual(result["hds"], 1.0)


class LayerThreeTests(unittest.TestCase):
    def test_kappa_is_one_for_identical_labels(self):
        self.assertEqual(evaluate_layer3.cohen_kappa([0, 1, 1], [0, 1, 1]), 1.0)

    def test_stratified_sample_is_deterministic_and_covers_strata(self):
        rows = []
        for model in ("a", "b"):
            for decision in ("clean", "hallucinated"):
                rows.append({"model": model, "combined_decision": decision,
                             "layer2_hallucination_probability": "",
                             "qid": f"{model}-{decision}"})
        first = prepare_layer3.stratified_sample(rows, 4, seed=42)
        second = prepare_layer3.stratified_sample(rows, 4, seed=42)
        self.assertEqual([row["qid"] for row in first],
                         [row["qid"] for row in second])
        self.assertEqual(len(first), 4)


if __name__ == "__main__":
    unittest.main()
