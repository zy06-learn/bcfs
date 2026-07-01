import assert from "node:assert/strict";
import { ASSIGNMENTS } from "../worker/survey_data.js";
import { validateAnnotatorSubmission, validateResponse } from "../worker/index.js";

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

function rowFor(annotator_id, eval_id) {
  return validateResponse({ ...validRow, annotator_id, eval_id }).response;
}

const completeRows = ASSIGNMENTS[annotatorId].map((assignedEvalId) => rowFor(annotatorId, assignedEvalId));
assert.equal(
  validateAnnotatorSubmission({ annotator_id: annotatorId, responses: completeRows }, completeRows).ok,
  true
);
assert.equal(
  validateAnnotatorSubmission({ annotator_id: annotatorId, responses: completeRows.slice(1) }, completeRows.slice(1)).ok,
  true
);
assert.equal(
  validateAnnotatorSubmission({ annotator_id: annotatorId, responses: completeRows }, [
    ...completeRows.slice(0, -1),
    completeRows[0],
  ]).ok,
  false
);
assert.equal(
  validateAnnotatorSubmission({ annotator_id: otherAnnotator, responses: completeRows }, completeRows).ok,
  false
);

console.log("worker_validation_ok");
