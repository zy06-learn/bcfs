# Human-Evaluation Survey Archive

This directory contains the static interface used for the paper's blind
pairwise evaluation: 100 CNN/DailyMail comparisons, 50 per backbone, assigned
once across 20 annotators. It is retained for protocol inspection and is not a
public results database.

The historical UI and database use a field named `currency`. The paper reports
the corresponding quality dimension as **coherence**. Because the stored schema
was not renamed before data collection, the release preserves the legacy field
name and documents the mismatch instead of silently asserting equivalence.

## Privacy Boundary

- `survey_items.json` contains the source, reference, and blinded candidate
  summaries shown to annotators.
- `assignments.json` and `annotator_links.csv` contain pseudonymous assignment
  identifiers.
- Raw participant responses, optional comments, and database exports are not
  committed.
- `worker/blind_key.js` maps A/B labels to systems and therefore belongs only in
  the backend Worker bundle, never in the Pages artifact.

## Local Static Preview

```bash
cd survey
python3 -m http.server 8000
```

Open `http://localhost:8000/?annotator=annotator_01`. Submission calls target
the configured Cloudflare Worker; use a disposable local Worker/D1 instance if
testing writes.

## Administrative API

Participant submission/status routes remain public for the archived workflow.
Routes that expose responses, corrected statistics, or re-evaluation data
require this header:

```text
Authorization: Bearer <SURVEY_ADMIN_TOKEN>
```

Protected routes are:

- `GET /api/export.csv`
- `GET /api/corrected/export.csv`
- `GET /api/corrected/stats`
- `GET /api/reeval/items`
- `POST /api/reeval/responses`
- `GET /api/reeval/export.csv`

Configure the Worker secret before deploying:

```bash
npx wrangler secret put SURVEY_ADMIN_TOKEN
```

Example authenticated export:

```bash
curl -fsSL \
  -H "Authorization: Bearer $SURVEY_ADMIN_TOKEN" \
  "https://bcfs-survey-api.zywang.workers.dev/api/export.csv" \
  -o outputs/human_eval_pairs/survey_responses_backend_raw.csv
```

Never place the token in a URL, tracked file, browser storage, log, or shared
annotator link. The re-evaluation page holds the entered token only in memory.

## Deployment

GitHub Pages deployment is manual through the `Deploy Human-Eval Survey Pages`
workflow. A push to `main` no longer deploys the survey automatically. Review
the Pages artifact before manually dispatching it.

Worker deployment and D1 migrations are separate, externally visible actions:

```bash
npm ci
npm run worker:test
npm run d1:migrate:remote
npm run worker:deploy
```

Run those commands only after reviewing the target Cloudflare account,
database, migration state, and secret configuration. Preparing this release
does not deploy either surface.
