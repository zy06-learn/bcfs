CREATE TABLE IF NOT EXISTS survey_responses (
  annotator_id TEXT NOT NULL,
  eval_id TEXT NOT NULL,
  task TEXT NOT NULL,
  sample_id TEXT NOT NULL,
  preference TEXT NOT NULL,
  A_consistency INTEGER NOT NULL,
  A_currency INTEGER NOT NULL,
  A_relevance INTEGER NOT NULL,
  A_clarity INTEGER NOT NULL,
  A_conciseness INTEGER NOT NULL,
  B_consistency INTEGER NOT NULL,
  B_currency INTEGER NOT NULL,
  B_relevance INTEGER NOT NULL,
  B_clarity INTEGER NOT NULL,
  B_conciseness INTEGER NOT NULL,
  comment TEXT NOT NULL DEFAULT '',
  submitted_at TEXT NOT NULL,
  PRIMARY KEY (annotator_id, eval_id)
);

CREATE INDEX IF NOT EXISTS idx_survey_responses_eval_id
  ON survey_responses (eval_id);

CREATE INDEX IF NOT EXISTS idx_survey_responses_submitted_at
  ON survey_responses (submitted_at);
