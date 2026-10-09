const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("question");
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");
const emptyEl = document.getElementById("empty-state");

function addMessage(kind, text) {
  if (emptyEl) emptyEl.hidden = true;
  const el = document.createElement("div");
  el.className = "msg " + kind;
  el.textContent = text; // textContent, never innerHTML: model output is untrusted
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function addSources(msgEl, sources) {
  if (!sources || !sources.length) return;
  const box = document.createElement("div");
  box.className = "sources";
  const title = document.createElement("strong");
  title.textContent = "Sources";
  const list = document.createElement("ul");
  for (const s of sources) {
    const li = document.createElement("li");
    li.textContent = s.page ? `${s.filename} — Page ${s.page}`
      : s.row ? `${s.filename} — ${s.row}` : s.filename;
    list.appendChild(li);
  }
  box.append(title, list);
  msgEl.appendChild(box);
}

function showLoading() {
  const el = document.createElement("div");
  el.className = "msg bot";
  el.innerHTML = '<span class="loading" aria-label="Thinking"><span></span><span></span><span></span></span>';
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

async function ask(question) {
  addMessage("user", question);
  const loadingEl = showLoading();
  sendBtn.disabled = true;
  inputEl.disabled = true;
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    loadingEl.remove();
    if (!res.ok) {
      let detail = "The assistant is temporarily unavailable. Please try again.";
      try { detail = (await res.json()).detail || detail; } catch (_) {}
      addMessage("bot error", typeof detail === "string" ? detail : "Invalid request.");
      return;
    }
    const data = await res.json();
    // A refusal means "not in the knowledge base", which is different from an error.
    const msg = addMessage(data.answered ? "bot" : "bot notice", data.answer);
    addSources(msg, data.sources);
  } catch (_) {
    loadingEl.remove();
    addMessage("bot error", "Could not reach the server. Check your connection and try again.");
  } finally {
    sendBtn.disabled = false;
    inputEl.disabled = false;
    inputEl.focus();
  }
}

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = inputEl.value.trim();
  if (!q) return;
  inputEl.value = "";
  ask(q);
});

document.getElementById("examples").addEventListener("click", (e) => {
  if (e.target.classList.contains("chip")) ask(e.target.textContent);
});

// ---- Theme toggle (initial theme is applied by the inline script in <head>) ----
const themeBtn = document.getElementById("theme-btn");

function syncThemeButton() {
  const dark = document.documentElement.getAttribute("data-theme") === "dark";
  themeBtn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  themeBtn.title = dark ? "Switch to light theme" : "Switch to dark theme";
}

themeBtn.addEventListener("click", () => {
  const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem("theme", next); } catch (_) {}
  syncThemeButton();
});
syncThemeButton();

clearBtn.addEventListener("click", () => {
  messagesEl.querySelectorAll(".msg").forEach((m) => m.remove());
  if (emptyEl) emptyEl.hidden = false;
});

// When shown inside the website's chat panel, Escape asks the page around us to close the panel.
if (window.self !== window.top) {
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") window.parent.postMessage({ type: "close-assistant" }, window.location.origin);
  });
}
