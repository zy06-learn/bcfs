# Human Evaluation Survey

This folder contains a static browser survey for pairwise summary evaluation.

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
- Annotations per item: 3
- Total item assignments: 300
- Min annotations per item: 3
- Max annotations per item: 3

## GitHub Pages Deployment

The workflow `.github/workflows/deploy-survey-pages.yml` deploys this folder as
the Pages artifact. If the workflow fails at `Configure Pages` with
`Resource not accessible by integration`, a repo owner/admin must enable Pages
once:

1. Open `https://github.com/zy06-learn/bcfs/settings/pages`.
2. Set Source to `GitHub Actions`.
3. Save the setting.
4. Re-run `Deploy Human-Eval Survey Pages` from the Actions tab, or push another
   small commit to `main`.

After it succeeds, the survey root is:

`https://zy06-learn.github.io/bcfs/`

## Response Collection

Ask every annotator to download and send back their response CSV. The public
site includes only blind survey items and assignments. Keep the original blind
key file private to the project owner.

Put returned CSV files into a local folder, for example `survey_responses/`, then run:

```bash
python3 scripts/merge_survey_responses.py \
  --responses_dir survey_responses \
  --blind_key outputs/human_eval_pairs/human_eval_100_pairs_blind_key.jsonl \
  --out_csv outputs/human_eval_pairs/merged_survey_responses.csv
```
