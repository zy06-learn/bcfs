const dimensions = [
  { id: "consistency", label: "Consistency" },
  { id: "currency", label: "Currency" },
  { id: "relevance", label: "Relevance" },
  { id: "clarity", label: "Clarity" },
  { id: "conciseness", label: "Conciseness" },
];

const API_BASE_URL = "https://bcfs-survey-api.zywang.workers.dev";

let items = [];
let assignments = {};
let currentAnnotator = "";
let reviewerId = "reviewer_01";
let assignedItems = [];
let currentIndex = 0;
let originalResponses = {};
let responses = {};
let isSubmitting = false;
let adminToken = "";

const $ = (id) => document.getElementById(id);

function adminHeaders(json = false) {
  const headers = { Authorization: `Bearer ${adminToken}` };
  if (json) headers["Content-Type"] = "application/json";
  return headers;
}

async function readApiResult(response) {
  const result = await response.json();
  if (!response.ok || !result.ok) {
    throw new Error(result.error || `HTTP ${response.status}`);
  }
  return result;
}

function getParams() {
  const params = new URLSearchParams(window.location.search);
  return {
    annotator: params.get("annotator") || "",
    reviewer: params.get("reviewer") || "reviewer_01",
  };
}

function setParams() {
  const url = new URL(window.location.href);
  if (currentAnnotator) url.searchParams.set("annotator", currentAnnotator);
  else url.searchParams.delete("annotator");
  if (reviewerId) url.searchParams.set("reviewer", reviewerId);
  window.history.replaceState({}, "", url);
}

function showStatus(message, timeout = 3500) {
  $("statusMessage").textContent = message;
  if (!message) return;
  window.clearTimeout(showStatus.timeoutId);
  showStatus.timeoutId = window.setTimeout(() => {
    $("statusMessage").textContent = "";
  }, timeout);
}

function labelPreference(value) {
  return {
    A_better: "A better",
    B_better: "B better",
    same: "Same",
    not_sure: "Not sure",
  }[value] || value || "";
}

function ratingLine(row, side) {
  const parts = dimensions.map((dimension) => {
    const value = row?.[`${side}_${dimension.id}`];
    return `${dimension.label}: ${value == null || value === "" ? "-" : value}`;
  });
  return `${side}: ${parts.join(" · ")}`;
}

function buildRatings(containerId, side) {
  const container = $(containerId);
  container.innerHTML = "";
  for (const dimension of dimensions) {
    const group = document.createElement("div");
    group.className = "rating-group";
    const label = document.createElement("span");
    label.className = "rating-label";
    label.textContent = dimension.label;
    const scale = document.createElement("div");
    scale.className = "scale";
    for (let score = 1; score <= 5; score += 1) {
      const option = document.createElement("label");
      option.textContent = score;
      const input = document.createElement("input");
      input.type = "radio";
      input.name = `${side}_${dimension.id}`;
      input.value = String(score);
      option.prepend(input);
      scale.appendChild(option);
    }
    group.append(label, scale);
    container.appendChild(group);
  }
}

function currentItem() {
  return assignedItems[currentIndex];
}

function isComplete(response) {
  return Boolean(response?.reeval_preference);
}

function completedCount() {
  return assignedItems.filter((item) => isComplete(responses[item.eval_id])).length;
}

function collectFormResponse() {
  const item = currentItem();
  if (!item) return;
  const formData = new FormData($("surveyForm"));
  const response = responses[item.eval_id] || {};
  response.reeval_preference = formData.get("reeval_preference") || "";
  for (const side of ["A", "B"]) {
    for (const dimension of dimensions) {
      response[`${side}_${dimension.id}`] = formData.get(`${side}_${dimension.id}`) || "";
    }
  }
  response.comment = $("comment").value.trim();
  responses[item.eval_id] = response;
  renderNav();
  renderProgress();
}

function applyResponseToForm(item) {
  const form = $("surveyForm");
  form.reset();
  const response = responses[item.eval_id] || {};
  if (response.reeval_preference) {
    const input = form.querySelector(`input[name="reeval_preference"][value="${response.reeval_preference}"]`);
    if (input) input.checked = true;
  }
  for (const side of ["A", "B"]) {
    for (const dimension of dimensions) {
      const value = response[`${side}_${dimension.id}`];
      if (!value) continue;
      const input = form.querySelector(`input[name="${side}_${dimension.id}"][value="${value}"]`);
      if (input) input.checked = true;
    }
  }
  $("comment").value = response.comment || "";
}

function renderCurrentItem() {
  const item = currentItem();
  if (!item) {
    $("surveyForm").classList.add("hidden");
    $("emptyState").classList.remove("hidden");
    $("emptyState").textContent = currentAnnotator
      ? "No original submitted items found for this annotator."
      : "Choose an annotator ID from the top right menu.";
    renderNav();
    renderProgress();
    return;
  }
  const original = originalResponses[item.eval_id] || {};
  $("emptyState").classList.add("hidden");
  $("surveyForm").classList.remove("hidden");
  $("itemMeta").textContent = `${item.task} · ${item.sample_id}`;
  $("itemTitle").textContent = `Submitted item ${currentIndex + 1} of ${assignedItems.length}`;
  $("originalChoice").textContent = [
    `Original preference: ${labelPreference(original.preference)}`,
    ratingLine(original, "A"),
    ratingLine(original, "B"),
    original.comment ? `Original comment: ${original.comment}` : "",
  ].filter(Boolean).join("\n");
  $("sourceText").textContent = item.source_text || "";
  $("referenceSummary").textContent = item.reference_summary || "";
  $("summaryA").textContent = item.summary_A || "";
  $("summaryB").textContent = item.summary_B || "";
  applyResponseToForm(item);
  $("prevButton").disabled = currentIndex === 0;
  $("nextButton").disabled = currentIndex === assignedItems.length - 1;
  renderNav();
  renderProgress();
}

function renderNav() {
  const nav = $("itemNav");
  nav.innerHTML = "";
  assignedItems.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = String(index + 1);
    if (index === currentIndex) button.classList.add("active");
    if (isComplete(responses[item.eval_id])) button.classList.add("done");
    button.addEventListener("click", () => {
      collectFormResponse();
      currentIndex = index;
      renderCurrentItem();
    });
    nav.appendChild(button);
  });
}

function renderProgress() {
  const done = completedCount();
  const total = assignedItems.length;
  $("progressCount").textContent = `${done} / ${total}`;
  $("progressBar").style.width = total ? `${(done / total) * 100}%` : "0";
  $("submitButton").disabled = !currentAnnotator || !assignedItems.length || isSubmitting;
  $("exportButton").disabled = !currentAnnotator || !assignedItems.length;
}

async function fetchReevalItems(annotator) {
  const url = `${API_BASE_URL}/api/reeval/items?annotator_id=${encodeURIComponent(annotator)}&reviewer_id=${encodeURIComponent(reviewerId)}`;
  const response = await fetch(url, { headers: adminHeaders() });
  return readApiResult(response);
}

async function chooseAnnotator(annotator) {
  collectFormResponse();
  currentAnnotator = annotator;
  reviewerId = $("reviewerInput").value.trim() || "reviewer_01";
  adminToken = $("adminTokenInput").value;
  setParams();
  showStatus("");
  originalResponses = {};
  responses = {};
  assignedItems = [];
  const itemById = new Map(items.map((item) => [item.eval_id, item]));
  if (annotator) {
    if (!adminToken) {
      $("annotatorLabel").textContent = "Enter the admin token before loading re-evaluation data.";
      $("annotatorSelect").value = annotator;
      renderCurrentItem();
      return;
    }
    const result = await fetchReevalItems(annotator);
    for (const row of result.original_responses || []) {
      originalResponses[row.eval_id] = row;
    }
    for (const row of result.reeval_responses || []) {
      responses[row.eval_id] = row;
    }
    assignedItems = (result.original_responses || [])
      .map((row) => itemById.get(row.eval_id))
      .filter(Boolean);
  }
  currentIndex = 0;
  $("annotatorLabel").textContent = annotator
    ? `${annotator}: ${assignedItems.length} submitted items to recheck`
    : "Select an annotator to re-evaluate submitted items.";
  $("annotatorSelect").value = annotator;
  renderCurrentItem();
}

function buildSubmissionRows() {
  return assignedItems.filter((item) => isComplete(responses[item.eval_id])).map((item) => ({
    original_annotator_id: currentAnnotator,
    eval_id: item.eval_id,
    reeval_preference: responses[item.eval_id].reeval_preference || "",
    A_consistency: responses[item.eval_id].A_consistency || "",
    A_currency: responses[item.eval_id].A_currency || "",
    A_relevance: responses[item.eval_id].A_relevance || "",
    A_clarity: responses[item.eval_id].A_clarity || "",
    A_conciseness: responses[item.eval_id].A_conciseness || "",
    B_consistency: responses[item.eval_id].B_consistency || "",
    B_currency: responses[item.eval_id].B_currency || "",
    B_relevance: responses[item.eval_id].B_relevance || "",
    B_clarity: responses[item.eval_id].B_clarity || "",
    B_conciseness: responses[item.eval_id].B_conciseness || "",
    comment: responses[item.eval_id].comment || "",
  }));
}

async function submitResponses() {
  collectFormResponse();
  const submissionRows = buildSubmissionRows();
  if (!submissionRows.length) {
    showStatus("No completed recheck items yet.", 8000);
    return;
  }
  isSubmitting = true;
  renderProgress();
  showStatus(`Saving ${submissionRows.length} recheck response${submissionRows.length === 1 ? "" : "s"}...`);
  try {
    const response = await fetch(`${API_BASE_URL}/api/reeval/responses`, {
      method: "POST",
      headers: adminHeaders(true),
      body: JSON.stringify({
        reviewer_id: reviewerId,
        annotator_id: currentAnnotator,
        responses: submissionRows,
      }),
    });
    const result = await response.json();
    if (!response.ok || !result.ok) {
      throw new Error(result.error || `HTTP ${response.status}`);
    }
    showStatus(`Saved ${result.saved_count} recheck response${result.saved_count === 1 ? "" : "s"}.`, 10000);
  } catch (error) {
    showStatus(`Save failed: ${error.message}`, 12000);
  } finally {
    isSubmitting = false;
    renderProgress();
  }
}

async function copyLink() {
  setParams();
  try {
    await navigator.clipboard.writeText(window.location.href);
    showStatus("Recheck link copied.");
  } catch {
    showStatus("Copy failed. Select and copy the browser URL manually.");
  }
}

async function exportCsv() {
  adminToken = $("adminTokenInput").value;
  if (!adminToken) {
    showStatus("Enter the admin token before exporting.", 8000);
    return;
  }
  try {
    const response = await fetch(`${API_BASE_URL}/api/reeval/export.csv`, {
      headers: adminHeaders(),
    });
    if (!response.ok) {
      await readApiResult(response);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "survey_reevaluations.csv";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    showStatus("Exported re-evaluation responses.");
  } catch (error) {
    showStatus(`Export failed: ${error.message}`, 12000);
  }
}

async function init() {
  const params = getParams();
  reviewerId = params.reviewer;
  $("reviewerInput").value = reviewerId;
  const [itemsResponse, assignmentsResponse] = await Promise.all([
    fetch("survey_items.json"),
    fetch("assignments.json"),
  ]);
  items = await itemsResponse.json();
  assignments = await assignmentsResponse.json();
  buildRatings("ratingsA", "A");
  buildRatings("ratingsB", "B");

  const select = $("annotatorSelect");
  select.innerHTML = '<option value="">Choose annotator</option>';
  Object.keys(assignments).forEach((annotator) => {
    const option = document.createElement("option");
    option.value = annotator;
    option.textContent = annotator;
    select.appendChild(option);
  });

  select.addEventListener("change", () => chooseAnnotator(select.value));
  $("adminTokenInput").addEventListener("change", () => chooseAnnotator(currentAnnotator));
  $("reviewerInput").addEventListener("change", () => chooseAnnotator(currentAnnotator));
  $("copyLinkButton").addEventListener("click", copyLink);
  $("submitButton").addEventListener("click", submitResponses);
  $("exportButton").addEventListener("click", exportCsv);
  $("prevButton").addEventListener("click", () => {
    collectFormResponse();
    currentIndex = Math.max(0, currentIndex - 1);
    renderCurrentItem();
  });
  $("nextButton").addEventListener("click", () => {
    collectFormResponse();
    currentIndex = Math.min(assignedItems.length - 1, currentIndex + 1);
    renderCurrentItem();
  });
  $("surveyForm").addEventListener("change", collectFormResponse);
  $("comment").addEventListener("input", collectFormResponse);

  await chooseAnnotator(params.annotator);
}

init().catch((error) => {
  $("emptyState").classList.remove("hidden");
  $("emptyState").textContent = `Failed to load recheck data: ${error.message}`;
});
