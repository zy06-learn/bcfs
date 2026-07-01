import { ASSIGNMENTS, SURVEY_ITEMS } from "./survey_data.js";

const ALLOWED_ORIGIN = "https://zy06-learn.github.io";
const LOCAL_ORIGIN_RE = /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/;

const DIMENSIONS = ["consistency", "currency", "relevance", "clarity", "conciseness"];
const PREFERENCES = new Set(["A_better", "B_better", "same", "not_sure"]);

const CSV_FIELDS = [
  "annotator_id",
  "eval_id",
  "task",
  "sample_id",
  "preference",
  "A_consistency",
  "A_currency",
  "A_relevance",
  "A_clarity",
  "A_conciseness",
  "B_consistency",
  "B_currency",
  "B_relevance",
  "B_clarity",
  "B_conciseness",
  "comment",
  "submitted_at",
];

const REEVAL_CSV_FIELDS = [
  "reviewer_id",
  "original_annotator_id",
  "eval_id",
  "task",
  "sample_id",
  "original_preference",
  "reeval_preference",
  "A_consistency",
  "A_currency",
  "A_relevance",
  "A_clarity",
  "A_conciseness",
  "B_consistency",
  "B_currency",
  "B_relevance",
  "B_clarity",
  "B_conciseness",
  "comment",
  "submitted_at",
];

function corsHeaders(request) {
  const origin = request.headers.get("Origin") || "";
  if (origin === ALLOWED_ORIGIN || LOCAL_ORIGIN_RE.test(origin)) {
    return {
      "Access-Control-Allow-Origin": origin,
      "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
      "Vary": "Origin",
    };
  }
  return {};
}

function jsonResponse(request, value, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...corsHeaders(request),
    },
  });
}

function isAllowedOrigin(request) {
  const origin = request.headers.get("Origin");
  return !origin || origin === ALLOWED_ORIGIN || LOCAL_ORIGIN_RE.test(origin);
}

function normalizePayload(payload) {
  if (Array.isArray(payload)) return { annotator_id: "", responses: payload };
  if (Array.isArray(payload?.responses)) return payload;
  return { annotator_id: payload?.annotator_id || "", responses: [payload] };
}

function toOptionalScore(value) {
  if (value == null || value === "") {
    return null;
  }
  const numberValue = Number(value);
  if (!Number.isInteger(numberValue) || numberValue < 1 || numberValue > 5) {
    return null;
  }
  return numberValue;
}

export function validateResponse(row) {
  const annotatorId = String(row?.annotator_id || "");
  const evalId = String(row?.eval_id || "");
  if (!annotatorId || !ASSIGNMENTS[annotatorId]) {
    return { ok: false, error: `unknown annotator_id: ${annotatorId}` };
  }
  if (!evalId || !SURVEY_ITEMS[evalId]) {
    return { ok: false, error: `unknown eval_id: ${evalId}` };
  }
  if (!ASSIGNMENTS[annotatorId].includes(evalId)) {
    return { ok: false, error: `${evalId} is not assigned to ${annotatorId}` };
  }
  if (!PREFERENCES.has(row?.preference)) {
    return { ok: false, error: `invalid preference for ${evalId}` };
  }

  const normalized = {
    annotator_id: annotatorId,
    eval_id: evalId,
    task: SURVEY_ITEMS[evalId].task,
    sample_id: SURVEY_ITEMS[evalId].sample_id,
    preference: row.preference,
    comment: String(row?.comment || "").slice(0, 5000),
    submitted_at: new Date().toISOString(),
  };

  for (const side of ["A", "B"]) {
    for (const dimension of DIMENSIONS) {
      const field = `${side}_${dimension}`;
      const score = toOptionalScore(row?.[field]);
      if (score == null && row?.[field] != null && row?.[field] !== "") {
        return { ok: false, error: `invalid ${field} for ${evalId}` };
      }
      normalized[field] = score;
    }
  }

  return { ok: true, response: normalized };
}

export function validateReevaluation(row) {
  const reviewerId = String(row?.reviewer_id || "reviewer_01").slice(0, 80);
  const originalAnnotatorId = String(row?.original_annotator_id || row?.annotator_id || "");
  const evalId = String(row?.eval_id || "");
  if (!reviewerId) {
    return { ok: false, error: "missing reviewer_id" };
  }
  if (!originalAnnotatorId || !ASSIGNMENTS[originalAnnotatorId]) {
    return { ok: false, error: `unknown original_annotator_id: ${originalAnnotatorId}` };
  }
  if (!evalId || !SURVEY_ITEMS[evalId]) {
    return { ok: false, error: `unknown eval_id: ${evalId}` };
  }
  if (!ASSIGNMENTS[originalAnnotatorId].includes(evalId)) {
    return { ok: false, error: `${evalId} is not assigned to ${originalAnnotatorId}` };
  }
  if (!PREFERENCES.has(row?.reeval_preference)) {
    return { ok: false, error: `invalid reeval_preference for ${evalId}` };
  }

  const normalized = {
    reviewer_id: reviewerId,
    original_annotator_id: originalAnnotatorId,
    eval_id: evalId,
    task: SURVEY_ITEMS[evalId].task,
    sample_id: SURVEY_ITEMS[evalId].sample_id,
    reeval_preference: row.reeval_preference,
    comment: String(row?.comment || "").slice(0, 5000),
    submitted_at: new Date().toISOString(),
  };

  for (const side of ["A", "B"]) {
    for (const dimension of DIMENSIONS) {
      const field = `${side}_${dimension}`;
      const score = toOptionalScore(row?.[field]);
      if (score == null && row?.[field] != null && row?.[field] !== "") {
        return { ok: false, error: `invalid ${field} for ${evalId}` };
      }
      normalized[field] = score;
    }
  }

  return { ok: true, response: normalized };
}

export function validateAnnotatorSubmission(payload, validated) {
  const annotatorId = String(payload?.annotator_id || "");
  if (!annotatorId || !ASSIGNMENTS[annotatorId]) {
    return { ok: false, error: `unknown annotator_id: ${annotatorId}` };
  }
  if (!validated.length) {
    return { ok: false, error: "empty_responses" };
  }
  if (validated.some((row) => row.annotator_id !== annotatorId)) {
    return { ok: false, error: "responses_must_match_payload_annotator_id" };
  }

  const expected = ASSIGNMENTS[annotatorId];
  const submitted = validated.map((row) => row.eval_id);
  const submittedSet = new Set(submitted);
  if (submittedSet.size !== submitted.length) {
    return { ok: false, error: "duplicate_eval_id_in_submission" };
  }

  const missing = expected.filter((evalId) => !submittedSet.has(evalId));
  const extra = submitted.filter((evalId) => !expected.includes(evalId));
  if (extra.length) {
    return {
      ok: false,
      error: "invalid_annotator_submission",
      extra,
      expected_count: expected.length,
      submitted_count: submitted.length,
    };
  }

  return {
    ok: true,
    annotator_id: annotatorId,
    missing,
    expected_count: expected.length,
    submitted_count: submitted.length,
  };
}

async function saveResponses(request, env) {
  if (!isAllowedOrigin(request)) {
    return jsonResponse(request, { ok: false, error: "origin_not_allowed" }, 403);
  }

  let payload;
  try {
    payload = normalizePayload(await request.json());
  } catch {
    return jsonResponse(request, { ok: false, error: "invalid_json" }, 400);
  }

  const validated = [];
  for (const row of payload.responses) {
    const result = validateResponse(row);
    if (!result.ok) {
      return jsonResponse(request, { ok: false, error: result.error }, 400);
    }
    validated.push(result.response);
  }

  const submission = validateAnnotatorSubmission(payload, validated);
  if (!submission.ok) {
    return jsonResponse(request, { ok: false, ...submission }, 400);
  }

  const sql = `
    INSERT INTO survey_responses (
      annotator_id, eval_id, task, sample_id, preference,
      A_consistency, A_currency, A_relevance, A_clarity, A_conciseness,
      B_consistency, B_currency, B_relevance, B_clarity, B_conciseness,
      comment, submitted_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(annotator_id, eval_id) DO UPDATE SET
      task = excluded.task,
      sample_id = excluded.sample_id,
      preference = excluded.preference,
      A_consistency = excluded.A_consistency,
      A_currency = excluded.A_currency,
      A_relevance = excluded.A_relevance,
      A_clarity = excluded.A_clarity,
      A_conciseness = excluded.A_conciseness,
      B_consistency = excluded.B_consistency,
      B_currency = excluded.B_currency,
      B_relevance = excluded.B_relevance,
      B_clarity = excluded.B_clarity,
      B_conciseness = excluded.B_conciseness,
      comment = excluded.comment,
      submitted_at = excluded.submitted_at
  `;

  const statements = validated.map((row) =>
    env.DB.prepare(sql).bind(...CSV_FIELDS.map((field) => row[field]))
  );
  await env.DB.batch(statements);

  return jsonResponse(request, {
    ok: true,
    saved_count: validated.length,
    annotator_id: submission.annotator_id,
    expected_count: submission.expected_count,
    submitted_count: submission.submitted_count,
    remaining_count: submission.missing.length,
  });
}

function csvEscape(value) {
  const text = value == null ? "" : String(value);
  return `"${text.replaceAll('"', '""')}"`;
}

async function exportCsv(request, env) {
  const { results } = await env.DB.prepare(
    "SELECT * FROM survey_responses ORDER BY annotator_id, eval_id"
  ).all();
  const lines = [CSV_FIELDS.join(",")];
  for (const row of results) {
    lines.push(CSV_FIELDS.map((field) => csvEscape(row[field])).join(","));
  }
  return new Response(`${lines.join("\n")}\n`, {
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": 'attachment; filename="survey_responses.csv"',
      ...corsHeaders(request),
    },
  });
}

async function submittedResponses(request, env) {
  const url = new URL(request.url);
  const annotatorId = String(url.searchParams.get("annotator_id") || "");
  if (!annotatorId || !ASSIGNMENTS[annotatorId]) {
    return jsonResponse(request, { ok: false, error: `unknown annotator_id: ${annotatorId}` }, 400);
  }
  const { results } = await env.DB.prepare(
    "SELECT eval_id, submitted_at FROM survey_responses WHERE annotator_id = ? ORDER BY eval_id"
  ).bind(annotatorId).all();
  return jsonResponse(request, {
    ok: true,
    annotator_id: annotatorId,
    submitted_eval_ids: results.map((row) => row.eval_id),
    submitted_count: results.length,
    expected_count: ASSIGNMENTS[annotatorId].length,
  });
}

async function reevalItems(request, env) {
  const url = new URL(request.url);
  const annotatorId = String(url.searchParams.get("annotator_id") || "");
  const reviewerId = String(url.searchParams.get("reviewer_id") || "reviewer_01").slice(0, 80);
  if (!annotatorId || !ASSIGNMENTS[annotatorId]) {
    return jsonResponse(request, { ok: false, error: `unknown annotator_id: ${annotatorId}` }, 400);
  }
  const original = await env.DB.prepare(
    "SELECT * FROM survey_responses WHERE annotator_id = ? ORDER BY eval_id"
  ).bind(annotatorId).all();
  const reeval = await env.DB.prepare(
    "SELECT * FROM survey_reevaluations WHERE reviewer_id = ? AND original_annotator_id = ? ORDER BY eval_id"
  ).bind(reviewerId, annotatorId).all();
  return jsonResponse(request, {
    ok: true,
    reviewer_id: reviewerId,
    annotator_id: annotatorId,
    original_responses: original.results,
    reeval_responses: reeval.results,
    original_count: original.results.length,
    reeval_count: reeval.results.length,
    expected_count: ASSIGNMENTS[annotatorId].length,
  });
}

async function saveReevaluations(request, env) {
  if (!isAllowedOrigin(request)) {
    return jsonResponse(request, { ok: false, error: "origin_not_allowed" }, 403);
  }

  let payload;
  try {
    payload = await request.json();
  } catch {
    return jsonResponse(request, { ok: false, error: "invalid_json" }, 400);
  }

  const responses = Array.isArray(payload?.responses) ? payload.responses : [];
  if (!responses.length) {
    return jsonResponse(request, { ok: false, error: "empty_responses" }, 400);
  }

  const validated = [];
  for (const row of responses) {
    const result = validateReevaluation({
      ...row,
      reviewer_id: payload.reviewer_id || row.reviewer_id,
      original_annotator_id: payload.annotator_id || payload.original_annotator_id || row.original_annotator_id,
    });
    if (!result.ok) {
      return jsonResponse(request, { ok: false, error: result.error }, 400);
    }
    const original = await env.DB.prepare(
      "SELECT preference FROM survey_responses WHERE annotator_id = ? AND eval_id = ?"
    ).bind(result.response.original_annotator_id, result.response.eval_id).first();
    if (!original) {
      return jsonResponse(request, {
        ok: false,
        error: `no original response for ${result.response.original_annotator_id}/${result.response.eval_id}`,
      }, 400);
    }
    validated.push({ ...result.response, original_preference: original.preference });
  }

  const sql = `
    INSERT INTO survey_reevaluations (
      reviewer_id, original_annotator_id, eval_id, task, sample_id,
      original_preference, reeval_preference,
      A_consistency, A_currency, A_relevance, A_clarity, A_conciseness,
      B_consistency, B_currency, B_relevance, B_clarity, B_conciseness,
      comment, submitted_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(reviewer_id, original_annotator_id, eval_id) DO UPDATE SET
      task = excluded.task,
      sample_id = excluded.sample_id,
      original_preference = excluded.original_preference,
      reeval_preference = excluded.reeval_preference,
      A_consistency = excluded.A_consistency,
      A_currency = excluded.A_currency,
      A_relevance = excluded.A_relevance,
      A_clarity = excluded.A_clarity,
      A_conciseness = excluded.A_conciseness,
      B_consistency = excluded.B_consistency,
      B_currency = excluded.B_currency,
      B_relevance = excluded.B_relevance,
      B_clarity = excluded.B_clarity,
      B_conciseness = excluded.B_conciseness,
      comment = excluded.comment,
      submitted_at = excluded.submitted_at
  `;

  const statements = validated.map((row) =>
    env.DB.prepare(sql).bind(...REEVAL_CSV_FIELDS.map((field) => row[field]))
  );
  await env.DB.batch(statements);

  return jsonResponse(request, {
    ok: true,
    saved_count: validated.length,
    reviewer_id: validated[0].reviewer_id,
    annotator_id: validated[0].original_annotator_id,
  });
}

async function exportReevalCsv(request, env) {
  const { results } = await env.DB.prepare(
    "SELECT * FROM survey_reevaluations ORDER BY reviewer_id, original_annotator_id, eval_id"
  ).all();
  const lines = [REEVAL_CSV_FIELDS.join(",")];
  for (const row of results) {
    lines.push(REEVAL_CSV_FIELDS.map((field) => csvEscape(row[field])).join(","));
  }
  return new Response(`${lines.join("\n")}\n`, {
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": 'attachment; filename="survey_reevaluations.csv"',
      ...corsHeaders(request),
    },
  });
}

async function health(request, env) {
  const row = await env.DB.prepare("SELECT COUNT(*) AS count FROM survey_responses").first();
  const completed = await env.DB.prepare(
    "SELECT annotator_id, COUNT(*) AS count FROM survey_responses GROUP BY annotator_id"
  ).all();
  const completedAnnotators = completed.results.filter(
    (result) => result.count === ASSIGNMENTS[result.annotator_id]?.length
  ).length;
  return jsonResponse(request, {
    ok: true,
    response_count: row?.count || 0,
    completed_annotator_count: completedAnnotators,
    assigned_response_count: Object.values(ASSIGNMENTS).reduce((sum, ids) => sum + ids.length, 0),
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(request) });
    }
    if (url.pathname === "/api/health" && request.method === "GET") {
      return health(request, env);
    }
    if (url.pathname === "/api/submitted" && request.method === "GET") {
      return submittedResponses(request, env);
    }
    if (url.pathname === "/api/reeval/items" && request.method === "GET") {
      return reevalItems(request, env);
    }
    if (url.pathname === "/api/reeval/responses" && request.method === "POST") {
      return saveReevaluations(request, env);
    }
    if (url.pathname === "/api/reeval/export.csv" && request.method === "GET") {
      return exportReevalCsv(request, env);
    }
    if (url.pathname === "/api/responses" && request.method === "POST") {
      return saveResponses(request, env);
    }
    if (url.pathname === "/api/export.csv" && request.method === "GET") {
      return exportCsv(request, env);
    }
    return jsonResponse(request, { ok: false, error: "not_found" }, 404);
  },
};
