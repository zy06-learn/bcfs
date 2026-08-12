import assert from "node:assert/strict";
import { ASSIGNMENTS } from "../worker/survey_data.js";
import worker, {
  authorizeAdmin,
  requiresAdmin,
  validateAnnotatorSubmission,
  validateReevaluation,
  validateResponse,
} from "../worker/index.js";

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
assert.equal(
  validateResponse({
    annotator_id: annotatorId,
    eval_id: evalId,
    preference: "A_better",
  }).ok,
  true
);
assert.equal(validateResponse({ ...validRow, eval_id: "missing" }).ok, false);
assert.equal(validateResponse({ ...validRow, preference: "invalid" }).ok, false);
assert.equal(validateResponse({ ...validRow, A_consistency: "6" }).ok, false);
assert.equal(
  validateReevaluation({
    reviewer_id: "reviewer_01",
    original_annotator_id: annotatorId,
    eval_id: evalId,
    reeval_preference: "B_better",
  }).ok,
  true
);
assert.equal(
  validateReevaluation({
    reviewer_id: "reviewer_01",
    original_annotator_id: annotatorId,
    eval_id: evalId,
    reeval_preference: "invalid",
  }).ok,
  false
);

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

const adminToken = "unit-test-placeholder";
const adminEnv = { SURVEY_ADMIN_TOKEN: adminToken };
const requestFor = (path, token, method = "GET") => new Request(`https://example.test${path}`, {
  method,
  headers: token == null ? {} : { Authorization: `Bearer ${token}` },
});

assert.equal(requiresAdmin("GET", "/api/export.csv"), true);
assert.equal(requiresAdmin("GET", "/api/corrected/stats"), true);
assert.equal(requiresAdmin("GET", "/api/reeval/items"), true);
assert.equal(requiresAdmin("POST", "/api/reeval/responses"), true);
assert.equal(requiresAdmin("GET", "/api/health"), false);
assert.equal(requiresAdmin("POST", "/api/responses"), false);
assert.deepEqual(
  authorizeAdmin(requestFor("/api/export.csv"), {}),
  { ok: false, status: 503, error: "admin_auth_not_configured" }
);
assert.deepEqual(
  authorizeAdmin(requestFor("/api/export.csv", "wrong"), adminEnv),
  { ok: false, status: 401, error: "admin_authorization_required" }
);
assert.deepEqual(
  authorizeAdmin(requestFor("/api/export.csv", `${adminToken}-suffix`), adminEnv),
  { ok: false, status: 401, error: "admin_authorization_required" }
);
assert.deepEqual(authorizeAdmin(requestFor("/api/export.csv", adminToken), adminEnv), { ok: true });

const unconfigured = await worker.fetch(requestFor("/api/export.csv"), {});
assert.equal(unconfigured.status, 503);
assert.equal((await unconfigured.json()).error, "admin_auth_not_configured");
const unauthorized = await worker.fetch(requestFor("/api/corrected/stats", "wrong"), adminEnv);
assert.equal(unauthorized.status, 401);
assert.equal((await unauthorized.json()).error, "admin_authorization_required");

let databaseWasRead = false;
const protectedEnv = {
  ...adminEnv,
  DB: {
    prepare() {
      databaseWasRead = true;
      return { all: async () => ({ results: [] }) };
    },
  },
};
const authorized = await worker.fetch(requestFor("/api/export.csv", adminToken), protectedEnv);
assert.equal(authorized.status, 200);
assert.equal(databaseWasRead, true);
assert.match(await authorized.text(), /^annotator_id,/);

console.log("worker_validation_ok");
