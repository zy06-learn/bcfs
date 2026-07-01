CREATE TABLE IF NOT EXISTS survey_reevaluations (
  reviewer_id TEXT NOT NULL,
  original_annotator_id TEXT NOT NULL,
  eval_id TEXT NOT NULL,
  task TEXT NOT NULL,
  sample_id TEXT NOT NULL,
  original_preference TEXT NOT NULL,
  reeval_preference TEXT NOT NULL,
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
  PRIMARY KEY (reviewer_id, original_annotator_id, eval_id)
);

CREATE INDEX IF NOT EXISTS idx_survey_reevaluations_original_annotator
  ON survey_reevaluations (original_annotator_id);

CREATE INDEX IF NOT EXISTS idx_survey_reevaluations_eval_id
  ON survey_reevaluations (eval_id);
