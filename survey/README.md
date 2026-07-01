# Human Evaluation Survey

This folder contains the public browser survey for pairwise summary evaluation.
Responses are submitted to the Cloudflare Worker API and can still be exported
as CSV from the browser as a backup.

Worker API:

- Health: `https://bcfs-survey-api.zywang.workers.dev/api/health`
- Export CSV: `https://bcfs-survey-api.zywang.workers.dev/api/export.csv`

## Local Test

```bash
cd survey
python3 -m http.server 8000
```

Open `http://localhost:8000/?annotator=annotator_01`.

## Annotator Links

Send one link to each annotator. Each annotator exports a CSV after finishing.

- `annotator_01`: `https://zy06-learn.github.io/bcfs/?annotator=annotator_01`
- `annotator_02`: `https://zy06-learn.github.io/bcfs/?annotator=annotator_02`
- `annotator_03`: `https://zy06-learn.github.io/bcfs/?annotator=annotator_03`
- `annotator_04`: `https://zy06-learn.github.io/bcfs/?annotator=annotator_04`
- `annotator_05`: `https://zy06-learn.github.io/bcfs/?annotator=annotator_05`
- `annotator_06`: `https://zy06-learn.github.io/bcfs/?annotator=annotator_06`
- `annotator_07`: `https://zy06-learn.github.io/bcfs/?annotator=annotator_07`
- `annotator_08`: `https://zy06-learn.github.io/bcfs/?annotator=annotator_08`
- `annotator_09`: `https://zy06-learn.github.io/bcfs/?annotator=annotator_09`
- `annotator_10`: `https://zy06-learn.github.io/bcfs/?annotator=annotator_10`
- `annotator_11`: `https://zy06-learn.github.io/bcfs/?annotator=annotator_11`
- `annotator_12`: `https://zy06-learn.github.io/bcfs/?annotator=annotator_12`
- `annotator_13`: `https://zy06-learn.github.io/bcfs/?annotator=annotator_13`
- `annotator_14`: `https://zy06-learn.github.io/bcfs/?annotator=annotator_14`
- `annotator_15`: `https://zy06-learn.github.io/bcfs/?annotator=annotator_15`
- `annotator_16`: `https://zy06-learn.github.io/bcfs/?annotator=annotator_16`
- `annotator_17`: `https://zy06-learn.github.io/bcfs/?annotator=annotator_17`
- `annotator_18`: `https://zy06-learn.github.io/bcfs/?annotator=annotator_18`
- `annotator_19`: `https://zy06-learn.github.io/bcfs/?annotator=annotator_19`
- `annotator_20`: `https://zy06-learn.github.io/bcfs/?annotator=annotator_20`

## Assignment Summary

- Items: 100
- Annotators: 20
- Annotations per item: 1
- Total item assignments: 100
- Min annotations per item: 1
- Max annotations per item: 1

## Response Collection

The primary collection path is automatic: annotators fill all assigned items and
click `Submit`. The fallback path is manual CSV export from the browser.

The public site includes only blind survey items and assignments. Keep the
original blind key file private to the project owner.

Download backend responses:

```bash
curl -fsSL "https://bcfs-survey-api.zywang.workers.dev/api/export.csv" \
  -o outputs/human_eval_pairs/survey_responses_backend_raw.csv
```

Decode A/B labels with the private blind key:

```bash
python3 scripts/merge_survey_responses.py \
  --responses_csv outputs/human_eval_pairs/survey_responses_backend_raw.csv \
  --blind_key outputs/human_eval_pairs/human_eval_100_pairs_blind_key.jsonl \
  --out_csv outputs/human_eval_pairs/merged_survey_responses.csv
```

If you collect fallback CSV files, put them into a local folder, for example
`survey_responses/`, then run:

```bash
python3 scripts/merge_survey_responses.py \
  --responses_dir survey_responses \
  --blind_key outputs/human_eval_pairs/human_eval_100_pairs_blind_key.jsonl \
  --out_csv outputs/human_eval_pairs/merged_survey_responses.csv
```
