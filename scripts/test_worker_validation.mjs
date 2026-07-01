import assert from "node:assert/strict";
import { ASSIGNMENTS } from "../worker/survey_data.js";
import { validateResponse } from "../worker/index.js";

const annotatorId = "annotator_01";
const evalId = ASSIGNMENTS[annotatorId][0];

const validRow = {
  annotator_id: annotatorId,
  eval_id: evalId,
  preference: "A_better",
  A_consistency: "5",
  A_currency: "4",
  A_relevance: "5",
  A_clarity: "4",
  A_conciseness: "3",
  B_consistency: "2",
  B_currency: "2",
  B_relevance: "3",
  B_clarity: "2",
  B_conciseness: "2",
  comment: "ok",
};

assert.equal(validateResponse(validRow).ok, true);
assert.equal(validateResponse({ ...validRow, eval_id: "missing" }).ok, false);
assert.equal(validateResponse({ ...validRow, preference: "invalid" }).ok, false);
assert.equal(validateResponse({ ...validRow, A_consistency: "6" }).ok, false);

const otherAnnotator = "annotator_02";
const otherEvalId = ASSIGNMENTS[otherAnnotator].find((candidate) => !ASSIGNMENTS[annotatorId].includes(candidate));
assert.equal(validateResponse({ ...validRow, eval_id: otherEvalId }).ok, false);

console.log("worker_validation_ok");
