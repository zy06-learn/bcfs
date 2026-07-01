CREATE TABLE IF NOT EXISTS survey_responses_new (
  annotator_id TEXT NOT NULL,
  eval_id TEXT NOT NULL,
  task TEXT NOT NULL,
  sample_id TEXT NOT NULL,
  preference TEXT NOT NULL,
  A_consistency INTEGER,
  A_currency INTEGER,
  A_relevance INTEGER,
  A_clarity INTEGER,
  A_conciseness INTEGER,
  B_consistency INTEGER,
  B_currency INTEGER,
  B_relevance INTEGER,
  B_clarity INTEGER,
  B_conciseness INTEGER,
  comment TEXT NOT NULL DEFAULT '',
  submitted_at TEXT NOT NULL,
  PRIMARY KEY (annotator_id, eval_id)
);

INSERT OR REPLACE INTO survey_responses_new (
  annotator_id, eval_id, task, sample_id, preference,
  A_consistency, A_currency, A_relevance, A_clarity, A_conciseness,
  B_consistency, B_currency, B_relevance, B_clarity, B_conciseness,
  comment, submitted_at
)
SELECT
  annotator_id, eval_id, task, sample_id, preference,
  A_consistency, A_currency, A_relevance, A_clarity, A_conciseness,
  B_consistency, B_currency, B_relevance, B_clarity, B_conciseness,
  comment, submitted_at
FROM survey_responses;

DROP TABLE survey_responses;

ALTER TABLE survey_responses_new RENAME TO survey_responses;

CREATE INDEX IF NOT EXISTS idx_survey_responses_eval_id
  ON survey_responses (eval_id);

CREATE INDEX IF NOT EXISTS idx_survey_responses_submitted_at
  ON survey_responses (submitted_at);
