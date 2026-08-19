import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.env.WAHM_ROOT ?? process.cwd();
const auditDir = path.join(root, "results", "judge_v2_audit_e9c5abb");
const trainingDir = path.join(root, "results", "judge_v2_e9c5abb");

const headerFormat = {
  fill: "#17324D",
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
};

async function saveWorkbook(workbook, outputPath) {
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
}

async function buildValidator(letter) {
  const csvPath = path.join(auditDir, `judge_v2_audit_validator_${letter}.csv`);
  const csvText = await fs.readFile(csvPath, "utf8");
  const workbook = await Workbook.fromCSV(csvText, { sheetName: "Audit" });
  const audit = workbook.worksheets.getItem("Audit");
  audit.showGridLines = false;
  audit.freezePanes.freezeRows(1);
  audit.freezePanes.freezeColumns(3);
  audit.getRange("A1:I1").format = headerFormat;
  audit.getRange("A2:I251").format.wrapText = true;
  audit.getRange("A2:I251").format.verticalAlignment = "top";
  audit.getRange("A1:A251").format.columnWidth = 14;
  audit.getRange("B1:B251").format.columnWidth = 15;
  audit.getRange("C1:C251").format.columnWidth = 13;
  audit.getRange("D1:E251").format.columnWidth = 42;
  audit.getRange("F1:F251").format.columnWidth = 58;
  audit.getRange("G1:H251").format.columnWidth = 25;
  audit.getRange("I1:I251").format.columnWidth = 34;
  audit.getRange("G2:G251").dataValidation = {
    rule: {
      type: "list",
      values: ["clean", "factual_hallucination", "degeneration"],
    },
  };
  audit.getRange("H2:H251").dataValidation = {
    rule: {
      type: "list",
      values: [
        "wrong_entity", "wrong_number_date", "contradiction",
        "unsupported_elaboration", "generic_imprecise",
        "instruction_mismatch", "degeneration", "other",
      ],
    },
  };
  audit.tables.add("A1:I251", true, `JudgeV2Audit${letter.toUpperCase()}`);

  const instructions = workbook.worksheets.add("Instructions");
  instructions.showGridLines = false;
  instructions.getRange("A1:F1").merge();
  instructions.getRange("A1").values = [["WAHM Judge v2 blinded audit"]];
  instructions.getRange("A1:F1").format = {
    fill: "#17324D",
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  instructions.getRange("A3:B9").values = [
    ["Goal", "Independently judge whether each answer is clean, factually hallucinated, or degeneration."],
    ["clean", "The answer is factually supported by the gold answer and does not add a false claim."],
    ["factual_hallucination", "The answer contains a factual error, contradiction, unsupported entity/number/date, or false elaboration."],
    ["degeneration", "The response is unusable because of blank/error output, foreign-script corruption, role leakage, code fences, or heavy repetition."],
    ["Error type", "Choose the closest error type only when the label is factual_hallucination or degeneration."],
    ["Independence", "Do not open the private key or the other validator workbook until both reviews are complete."],
    ["Completion", "Fill human_label for all 250 rows. Add comments whenever the decision may require adjudication."],
  ];
  instructions.getRange("A3:A9").format = {
    fill: "#DCE6F1",
    font: { bold: true, color: "#17324D" },
    verticalAlignment: "top",
  };
  instructions.getRange("B3:B9").format = {
    wrapText: true,
    verticalAlignment: "top",
  };
  instructions.getRange("A3:B9").format.borders = {
    preset: "all", style: "thin", color: "#B7C9D6",
  };
  instructions.getRange("A1:A9").format.columnWidth = 25;
  instructions.getRange("B1:B9").format.columnWidth = 95;
  instructions.getRange("A1:F1").format.rowHeight = 34;

  const outputPath = path.join(auditDir, `judge_v2_validator_${letter}.xlsx`);
  const preview = await workbook.render({
    sheetName: "Audit", range: "A1:I18", scale: 1, format: "png",
  });
  await fs.writeFile(
    path.join(auditDir, `judge_v2_validator_${letter}_preview.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
  await saveWorkbook(workbook, outputPath);
  return workbook;
}

async function buildTrainingDiagnostics() {
  const csvText = await fs.readFile(path.join(trainingDir, "epoch_metrics.csv"), "utf8");
  const workbook = await Workbook.fromCSV(csvText, { sheetName: "Epoch Metrics" });
  const epochs = workbook.worksheets.getItem("Epoch Metrics");
  epochs.showGridLines = false;
  epochs.freezePanes.freezeRows(1);
  epochs.getRange("A1:J1").format = headerFormat;
  epochs.getRange("A1:J4").format.columnWidth = 22;
  epochs.getRange("A2:A4").setNumberFormat("0");
  epochs.getRange("B2:J4").setNumberFormat("0.0000");
  epochs.tables.add("A1:J4", true, "EpochMetrics");

  const run = JSON.parse(await fs.readFile(path.join(trainingDir, "run_summary.json"), "utf8"));
  const config = JSON.parse(await fs.readFile(path.join(trainingDir, "training_config.json"), "utf8"));
  const summary = workbook.worksheets.add("Summary");
  summary.showGridLines = false;
  summary.getRange("A1:M1").merge();
  summary.getRange("A1").values = [["WAHM Judge v2 — training diagnostics"]];
  summary.getRange("A1:M1").format = {
    fill: "#17324D",
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  summary.getRange("A3:B11").values = [
    ["Checkpoint commit", "e9c5abbf2048c7e2d5762036b722aebc154e7295"],
    ["Base model", config.base_model],
    ["Grouped split", `${config.n_train} train / ${config.n_validation} validation / ${config.n_test} test`],
    ["Decision threshold", run.decision_threshold],
    ["Validation F1", run.validation_metrics.f1],
    ["Validation ROC-AUC", run.validation_metrics.roc_auc],
    ["Test F1", run.test_metrics.f1],
    ["Test ROC-AUC", run.test_metrics.roc_auc],
    ["Weights SHA-256", run.model_weights_sha256],
  ];
  summary.getRange("A3:A11").format = {
    fill: "#DCE6F1", font: { bold: true, color: "#17324D" },
  };
  summary.getRange("A3:B11").format.borders = {
    preset: "all", style: "thin", color: "#B7C9D6",
  };
  summary.getRange("A1:A23").format.columnWidth = 25;
  summary.getRange("B1:B23").format.columnWidth = 70;
  summary.getRange("B6:B10").setNumberFormat("0.0000");

  summary.getRange("A14:C17").values = [
    ["Epoch", "Train loss", "Validation loss"],
    [1, 0.4821, 0.5040968657],
    [2, 0.3108, 0.4650818408],
    [3, 0.2339, 0.4729795456],
  ];
  summary.getRange("A20:C23").values = [
    ["Epoch", "Validation F1", "Validation ROC-AUC"],
    [1, 0.774566, 0.842102],
    [2, 0.817372, 0.879936],
    [3, 0.831661, 0.88561],
  ];
  summary.getRange("A14:C14").format = headerFormat;
  summary.getRange("A20:C20").format = headerFormat;
  summary.getRange("B15:C17").setNumberFormat("0.0000");
  summary.getRange("B21:C23").setNumberFormat("0.0000");
  summary.getRange("A14:C17").format.borders = {
    preset: "all", style: "thin", color: "#B7C9D6",
  };
  summary.getRange("A20:C23").format.borders = {
    preset: "all", style: "thin", color: "#B7C9D6",
  };

  const lossChart = summary.charts.add("line", summary.getRange("A14:C17"));
  lossChart.title = "Training vs validation loss";
  lossChart.hasLegend = true;
  lossChart.xAxis = { axisType: "textAxis" };
  lossChart.yAxis = { numberFormatCode: "0.00", min: 0, max: 0.6 };
  lossChart.setPosition("E3", "M17");

  const qualityChart = summary.charts.add("line", summary.getRange("A20:C23"));
  qualityChart.title = "Validation quality by epoch";
  qualityChart.hasLegend = true;
  qualityChart.xAxis = { axisType: "textAxis" };
  qualityChart.yAxis = { numberFormatCode: "0.00", min: 0.7, max: 0.95 };
  qualityChart.setPosition("E19", "M33");

  const preview = await workbook.render({
    sheetName: "Summary", range: "A1:M33", scale: 1, format: "png",
  });
  await fs.writeFile(
    path.join(trainingDir, "training_diagnostics_preview.png"),
    new Uint8Array(await preview.arrayBuffer()),
  );
  await saveWorkbook(workbook, path.join(trainingDir, "training_diagnostics.xlsx"));
  return workbook;
}

const workbooks = [
  await buildValidator("a"),
  await buildValidator("b"),
  await buildTrainingDiagnostics(),
];
for (const workbook of workbooks) {
  const inspection = await workbook.inspect({
    kind: "sheet,table", maxChars: 3000, tableMaxRows: 3, tableMaxCols: 10,
  });
  console.log(inspection.ndjson);
}
