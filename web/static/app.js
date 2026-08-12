const question = document.querySelector("#question");
const topK = document.querySelector("#top-k");
const askButton = document.querySelector("#ask-button");
const result = document.querySelector("#result");
const answer = document.querySelector("#answer");
const contexts = document.querySelector("#contexts");
const error = document.querySelector("#error");

function renderAnswerWithCitations(text, retrievedContexts) {
  const citations = retrievedContexts.map((context, index) => ({
    raw: `[${context.source}#${context.chunk_id}]`,
    label: `근거 ${index + 1}`,
  }));
  answer.replaceChildren();
  if (!citations.length) {
    answer.textContent = text;
    return;
  }

  const citationByRaw = new Map(citations.map((citation) => [citation.raw, citation]));
  const pattern = new RegExp(
    citations
      .map((citation) => citation.raw.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
      .join("|"),
    "g",
  );

  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    answer.append(document.createTextNode(text.slice(cursor, match.index)));
    const citation = document.createElement("span");
    citation.className = "citation";
    citation.textContent = citationByRaw.get(match[0]).label;
    citation.title = match[0].slice(1, -1);
    answer.append(citation);
    cursor = match.index + match[0].length;
  }
  answer.append(document.createTextNode(text.slice(cursor)));
}

function resizeQuestion() {
  question.style.height = "auto";
  question.style.height = `${Math.min(question.scrollHeight, 220)}px`;
  question.style.overflowY = question.scrollHeight > 220 ? "auto" : "hidden";
}

question.addEventListener("input", resizeQuestion);

document.querySelectorAll(".example").forEach((button) => {
  button.addEventListener("click", () => {
    question.value = button.textContent.trim();
    resizeQuestion();
    question.focus();
  });
});

askButton.addEventListener("click", async () => {
  const value = question.value.trim();
  if (!value) {
    error.textContent = "질문을 입력해 주세요.";
    return;
  }

  error.textContent = "";
  result.hidden = true;
  askButton.disabled = true;
  const loadingLabel = askButton.firstElementChild;
  let dotCount = 1;
  loadingLabel.textContent = "문서를 검색하고 있습니다.";
  const loadingAnimation = window.setInterval(() => {
    dotCount = (dotCount % 3) + 1;
    loadingLabel.textContent = `문서를 검색하고 있습니다${".".repeat(dotCount)}`;
  }, 450);

  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: value, top_k: Number(topK.value) }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "질의 처리에 실패했습니다.");

    renderAnswerWithCitations(data.answer, data.contexts);
    contexts.replaceChildren();
    data.contexts.forEach((context, index) => {
      const details = document.createElement("details");
      details.className = "context";
      const summary = document.createElement("summary");
      summary.textContent = `근거 ${index + 1} · ${context.source}`;
      const meta = document.createElement("span");
      meta.className = "context-meta";
      meta.textContent = `${context.chunk_id} · similarity ${context.similarity.toFixed(4)}`;
      const text = document.createElement("pre");
      text.textContent = context.text;
      details.append(summary, meta, text);
      contexts.append(details);
    });
    result.hidden = false;
    result.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (reason) {
    error.textContent = reason.message;
  } finally {
    window.clearInterval(loadingAnimation);
    askButton.disabled = false;
    loadingLabel.textContent = "근거 기반 답변 생성";
  }
});
