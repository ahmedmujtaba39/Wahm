import unittest

import numpy as np

from translation_metrics_v2 import backtrans_cosine_similarities


class FakeSentenceEncoder:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        vectors = []
        for text in texts:
            vector = np.array([len(text), sum(map(ord, text)) % 997], dtype=float)
            if normalize_embeddings:
                vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.stack(vectors)


class BacktranslationCosineTests(unittest.TestCase):
    def test_identical_text_has_cosine_near_one(self):
        text = "ما هي عاصمة المملكة العربية السعودية؟"
        score = backtrans_cosine_similarities(
            [text], [text], model=FakeSentenceEncoder()
        )[0]
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_empty_backtranslation_has_no_score(self):
        self.assertEqual(
            backtrans_cosine_similarities(["سؤال"], [""]), [None]
        )


if __name__ == "__main__":
    unittest.main()
