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
let assignedItems = [];
let currentIndex = 0;
let responses = {};
let isSubmitting = false;
let submittedEvalIds = new Set();

const $ = (id) => document.getElementById(id);

function storageKey() {
  return `summary_eval_${currentAnnotator}`;
}

function csvEscape(value) {
  const text = value == null ? "" : String(value);
  return `"${text.replaceAll('"', '""')}"`;
}

function getAnnotatorFromUrl() {
  const params = new URLSearchParams(window.location.search);
  return params.get("annotator") || "";
}

function setAnnotatorInUrl(annotator) {
  const url = new URL(window.location.href);
  if (annotator) {
    url.searchParams.set("annotator", annotator);
  } else {
    url.searchParams.delete("annotator");
  }
  window.history.replaceState({}, "", url);
}

function loadResponses() {
  if (!currentAnnotator) return {};
  try {
    return JSON.parse(localStorage.getItem(storageKey()) || "{}");
  } catch {
    return {};
  }
}

function saveResponses() {
  if (!currentAnnotator) return;
  localStorage.setItem(storageKey(), JSON.stringify(responses));
}

function showStatus(message, timeout = 3500) {
  $("statusMessage").textContent = message;
  if (!message) return;
  window.clearTimeout(showStatus.timeoutId);
  showStatus.timeoutId = window.setTimeout(() => {
    $("statusMessage").textContent = "";
  }, timeout);
}

function isComplete(response) {
  return Boolean(response?.preference);
}

function completedCount() {
  return assignedItems.filter((item) => isComplete(responses[item.eval_id])).length;
}

function firstIncompleteIndex() {
  return assignedItems.findIndex((item) => !isComplete(responses[item.eval_id]));
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

function collectFormResponse() {
  const item = currentItem();
  if (!item) return;
  const form = $("surveyForm");
  const formData = new FormData(form);
  const response = responses[item.eval_id] || {};
  response.preference = formData.get("preference") || "";
  for (const side of ["A", "B"]) {
    for (const dimension of dimensions) {
      response[`${side}_${dimension.id}`] = formData.get(`${side}_${dimension.id}`) || "";
    }
  }
  response.comment = $("comment").value.trim();
  response.updated_at = new Date().toISOString();
  responses[item.eval_id] = response;
  saveResponses();
  renderNav();
  renderProgress();
}

function applyResponseToForm(item) {
  const form = $("surveyForm");
  form.reset();
  const response = responses[item.eval_id] || {};
  if (response.preference) {
    const input = form.querySelector(`input[name="preference"][value="${response.preference}"]`);
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
      ? "All assigned items have been submitted for this annotator."
      : "No assigned items found for this annotator.";
    return;
  }
  $("emptyState").classList.add("hidden");
  $("surveyForm").classList.remove("hidden");
  $("itemMeta").textContent = `${item.task} · ${item.sample_id}`;
  $("itemTitle").textContent = `Item ${currentIndex + 1} of ${assignedItems.length}`;
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
  $("submitButton").disabled = !currentAnnotator || isSubmitting;
}

async function fetchSubmittedEvalIds(annotator) {
  if (!annotator) return new Set();
  const response = await fetch(`${API_BASE_URL}/api/submitted?annotator_id=${encodeURIComponent(annotator)}`);
  if (!response.ok) {
    throw new Error(`failed to load submitted items: HTTP ${response.status}`);
  }
  const result = await response.json();
  if (!result.ok) {
    throw new Error(result.error || "failed to load submitted items");
  }
  return new Set(result.submitted_eval_ids || []);
}

async function chooseAnnotator(annotator) {
  currentAnnotator = annotator;
  setAnnotatorInUrl(annotator);
  showStatus("");
  responses = loadResponses();
  submittedEvalIds = new Set();
  const itemById = new Map(items.map((item) => [item.eval_id, item]));
  if (annotator) {
    try {
      submittedEvalIds = await fetchSubmittedEvalIds(annotator);
    } catch (error) {
      showStatus(`${error.message}. Showing local remaining items only.`, 10000);
    }
  }
  assignedItems = (assignments[annotator] || [])
    .filter((id) => !submittedEvalIds.has(id))
    .map((id) => itemById.get(id))
    .filter(Boolean);
  currentIndex = 0;
  $("annotatorLabel").textContent = annotator
    ? `${annotator}: ${assignedItems.length} remaining items`
    : "Select an annotator to begin.";
  $("annotatorSelect").value = annotator;
  renderCurrentItem();
}

function exportCsv() {
  collectFormResponse();
  if (!currentAnnotator) return;
  const done = completedCount();
  if (done < assignedItems.length) {
    const remaining = assignedItems.length - done;
    const shouldExport = window.confirm(
      `${remaining} assigned item${remaining === 1 ? "" : "s"} still incomplete. Export anyway?`
    );
    if (!shouldExport) {
      showStatus("Export cancelled. Finish the remaining items first.");
      return;
    }
  }
  const fields = [
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
    "updated_at",
  ];
  const lines = [fields.join(",")];
  for (const item of assignedItems) {
    const response = responses[item.eval_id] || {};
    const row = {
      annotator_id: currentAnnotator,
      eval_id: item.eval_id,
      task: item.task,
      sample_id: item.sample_id,
      ...response,
    };
    lines.push(fields.map((field) => csvEscape(row[field])).join(","));
  }
  const blob = new Blob([`${lines.join("\n")}\n`], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${currentAnnotator}_summary_eval_responses.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  showStatus(`Exported ${assignedItems.length} rows for ${currentAnnotator}.`);
}

function buildSubmissionRows() {
  return assignedItems.filter((item) => isComplete(responses[item.eval_id])).map((item) => {
    const response = responses[item.eval_id] || {};
    return {
      annotator_id: currentAnnotator,
      eval_id: item.eval_id,
      task: item.task,
      sample_id: item.sample_id,
      preference: response.preference || "",
      A_consistency: response.A_consistency || "",
      A_currency: response.A_currency || "",
      A_relevance: response.A_relevance || "",
      A_clarity: response.A_clarity || "",
      A_conciseness: response.A_conciseness || "",
      B_consistency: response.B_consistency || "",
      B_currency: response.B_currency || "",
      B_relevance: response.B_relevance || "",
      B_clarity: response.B_clarity || "",
      B_conciseness: response.B_conciseness || "",
      comment: response.comment || "",
    };
  });
}

async function submitResponses() {
  collectFormResponse();
  if (!currentAnnotator) return;
  const submissionRows = buildSubmissionRows();
  if (!submissionRows.length) {
    const incompleteIndex = firstIncompleteIndex();
    if (incompleteIndex >= 0) {
      currentIndex = incompleteIndex;
      renderCurrentItem();
    }
    showStatus("Cannot submit yet: no completed items. Finish this pair first.", 8000);
    return;
  }

  isSubmitting = true;
  $("submitButton").disabled = true;
  showStatus(`Submitting ${submissionRows.length} completed response${submissionRows.length === 1 ? "" : "s"}...`);
  try {
    const response = await fetch(`${API_BASE_URL}/api/responses`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        annotator_id: currentAnnotator,
        responses: submissionRows,
      }),
    });
    const result = await response.json();
    if (!response.ok || !result.ok) {
      throw new Error(result.error || `HTTP ${response.status}`);
    }
    localStorage.setItem(`${storageKey()}_submitted_at`, new Date().toISOString());
    const submittedIds = new Set(submissionRows.map((row) => row.eval_id));
    assignedItems = assignedItems.filter((item) => !submittedIds.has(item.eval_id));
    currentIndex = Math.min(currentIndex, Math.max(assignedItems.length - 1, 0));
    const remaining = assignedItems.length;
    $("annotatorLabel").textContent = `${currentAnnotator}: ${assignedItems.length} remaining items`;
    renderCurrentItem();
    if (remaining) {
      showStatus(
        `Submitted ${result.saved_count} completed response${result.saved_count === 1 ? "" : "s"}. ${remaining} item${remaining === 1 ? "" : "s"} still incomplete.`,
        10000
      );
    } else {
      showStatus(`Submitted all ${result.saved_count} responses. You can close this page.`, 10000);
    }
  } catch (error) {
    showStatus(`Submit failed: ${error.message}. Use Export CSV as backup.`, 12000);
  } finally {
    isSubmitting = false;
    renderProgress();
  }
}

async function copyLink() {
  if (!currentAnnotator) return;
  try {
    await navigator.clipboard.writeText(window.location.href);
    showStatus("Annotator link copied.");
  } catch {
    showStatus("Copy failed. Select and copy the browser URL manually.");
  }
}

async function init() {
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

  await chooseAnnotator(getAnnotatorFromUrl());
}

init().catch((error) => {
  $("emptyState").classList.remove("hidden");
  $("emptyState").textContent = `Failed to load survey data: ${error.message}`;
});
