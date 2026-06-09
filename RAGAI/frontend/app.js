/* ═══════════════════════════════════════════════════════════════════
   RAG AI Agent — Frontend Application
   Full SPA: SSE streaming, document management, session handling
   ═══════════════════════════════════════════════════════════════════ */

const API = window.location.origin;

// ── State ──────────────────────────────────────────────────────────
const state = {
  sessionId: null,
  isStreaming: false,
  documents: [],
};

// ── DOM Elements ───────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const els = {
  sidebar:       $("#sidebar"),
  sidebarToggle: $("#sidebarToggle"),
  mobileMenuBtn: $("#mobileMenuBtn"),
  statusDot:     $("#statusDot"),
  statusText:    $("#statusText"),
  statusDocs:    $("#statusDocs"),
  newChatBtn:    $("#newChatBtn"),
  refreshDocsBtn:$("#refreshDocsBtn"),
  uploadZone:    $("#uploadZone"),
  fileInput:     $("#fileInput"),
  uploadProgress:$("#uploadProgress"),
  progressFill:  $("#progressFill"),
  progressLabel: $("#progressLabel"),
  docList:       $("#docList"),
  docEmpty:      $("#docEmpty"),
  resetKbBtn:    $("#resetKbBtn"),
  chatArea:      $("#chatArea"),
  welcomeScreen: $("#welcomeScreen"),
  messages:      $("#messages"),
  chatInput:     $("#chatInput"),
  charCount:     $("#charCount"),
  sendBtn:       $("#sendBtn"),
  clearChatBtn:  $("#clearChatBtn"),
  headerTitle:   $("#headerTitle"),
  headerSub:     $("#headerSub"),
  toastContainer:$("#toastContainer"),
  confirmModal:  $("#confirmModal"),
  modalTitle:    $("#modalTitle"),
  modalBody:     $("#modalBody"),
  modalCancel:   $("#modalCancel"),
  modalConfirm:  $("#modalConfirm"),
  suggestionChips:$("#suggestionChips"),
};

// ═══════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════

function toast(message, type = "info", duration = 4000) {
  const t = document.createElement("div");
  t.className = `toast ${type}`;
  const icons = { success: "✓", error: "✕", info: "ℹ" };
  t.innerHTML = `<span>${icons[type] || "ℹ"}</span><span class="toast-msg">${escapeHtml(message)}</span>`;
  els.toastContainer.appendChild(t);
  setTimeout(() => {
    t.style.opacity = "0";
    t.style.transform = "translateY(10px)";
    t.style.transition = "all 0.3s ease";
    setTimeout(() => t.remove(), 300);
  }, duration);
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function formatTime(ts) {
  try {
    const d = ts ? new Date(ts) : new Date();
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch { return ""; }
}

function getFileIcon(filename) {
  const ext = filename.split(".").pop().toLowerCase();
  const icons = {
    pdf: "📕", docx: "📘", doc: "📘", txt: "📝", md: "📝",
    csv: "📊", html: "🌐", htm: "🌐",
  };
  return { icon: icons[ext] || "📄", ext };
}

/** Minimal markdown → HTML.  Handles bold, italic, code, headings, lists, blockquotes, links */
function renderMarkdown(text) {
  if (!text) return "";
  let html = escapeHtml(text);

  // Code blocks (``` ... ```)
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) =>
    `<pre><code class="lang-${lang}">${code.trim()}</code></pre>`
  );

  // Inline code
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

  // Headings
  html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
  html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");

  // Bold & italic
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

  // Blockquotes
  html = html.replace(/^&gt; (.+)$/gm, "<blockquote>$1</blockquote>");

  // Unordered lists
  html = html.replace(/^[\-\*] (.+)$/gm, "<li>$1</li>");
  html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>");

  // Ordered lists
  html = html.replace(/^\d+\. (.+)$/gm, "<li>$1</li>");

  // Links
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

  // Paragraphs (double newlines)
  html = html.replace(/\n{2,}/g, "</p><p>");

  // Single newlines → <br>
  html = html.replace(/\n/g, "<br>");

  return `<p>${html}</p>`;
}

// ═══════════════════════════════════════════════════════════════════
// API CALLS
// ═══════════════════════════════════════════════════════════════════

async function apiGet(path) {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

async function apiPost(path, body, isJson = true) {
  const opts = { method: "POST" };
  if (isJson) {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  } else {
    opts.body = body; // FormData
  }
  const res = await fetch(`${API}${path}`, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

async function apiDelete(path) {
  const res = await fetch(`${API}${path}`, { method: "DELETE" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

// ═══════════════════════════════════════════════════════════════════
// HEALTH CHECK
// ═══════════════════════════════════════════════════════════════════

async function checkHealth() {
  els.statusDot.className = "status-dot pulsing";
  els.statusText.textContent = "Connecting...";
  try {
    const data = await apiGet("/health");
    const ok = data.status === "healthy";
    els.statusDot.className = `status-dot ${ok ? "ok" : "warn"}`;
    els.statusText.textContent = ok ? "Connected" : "Degraded";
    els.statusDocs.textContent = `${data.document_count} chunks`;
    els.statusDocs.hidden = false;
    return data;
  } catch (err) {
    els.statusDot.className = "status-dot error";
    els.statusText.textContent = "Offline";
    els.statusDocs.hidden = true;
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════
// DOCUMENTS
// ═══════════════════════════════════════════════════════════════════

async function loadDocuments() {
  try {
    state.documents = await apiGet("/api/documents/");
    renderDocList();
  } catch (err) {
    console.error("Failed to load documents:", err);
  }
}

function renderDocList() {
  // Remove existing doc-items
  els.docList.querySelectorAll(".doc-item").forEach((el) => el.remove());

  if (state.documents.length === 0) {
    els.docEmpty.hidden = false;
    return;
  }
  els.docEmpty.hidden = true;

  state.documents.forEach((doc) => {
    const { icon, ext } = getFileIcon(doc.filename);
    const item = document.createElement("div");
    item.className = "doc-item";
    item.innerHTML = `
      <div class="doc-icon ${ext}">${icon}</div>
      <div class="doc-info">
        <div class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
        <div class="doc-meta">${doc.total_chunks} chunks · ${doc.doc_id.slice(0, 8)}</div>
      </div>
      <button class="doc-delete" data-id="${doc.doc_id}" title="Delete document">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
      </button>
    `;
    item.querySelector(".doc-delete").addEventListener("click", (e) => {
      e.stopPropagation();
      confirmAction(
        "Delete Document",
        `Remove "${doc.filename}" from the knowledge base? This cannot be undone.`,
        () => deleteDocument(doc.doc_id)
      );
    });
    els.docList.appendChild(item);
  });
}

async function uploadFiles(files) {
  if (!files.length) return;
  els.uploadProgress.hidden = false;
  els.progressFill.style.width = "0%";
  els.progressLabel.textContent = `Uploading ${files.length} file(s)...`;

  const formData = new FormData();
  for (const f of files) formData.append("files", f);

  // Simulate progress
  let pct = 0;
  const timer = setInterval(() => {
    pct = Math.min(pct + Math.random() * 15, 90);
    els.progressFill.style.width = `${pct}%`;
  }, 300);

  try {
    const result = await apiPost("/api/documents/upload", formData, false);
    clearInterval(timer);
    els.progressFill.style.width = "100%";

    if (result.documents?.length) {
      toast(`Uploaded ${result.documents.length} document(s) successfully`, "success");
    }
    if (result.errors?.length) {
      result.errors.forEach((e) => toast(e, "error"));
    }

    await loadDocuments();
    await checkHealth();
  } catch (err) {
    clearInterval(timer);
    toast(`Upload failed: ${err.message}`, "error");
  } finally {
    setTimeout(() => {
      els.uploadProgress.hidden = true;
      els.progressFill.style.width = "0%";
    }, 1000);
  }
}

async function deleteDocument(docId) {
  try {
    await apiDelete(`/api/documents/${docId}`);
    toast("Document deleted", "success");
    await loadDocuments();
    await checkHealth();
  } catch (err) {
    toast(`Delete failed: ${err.message}`, "error");
  }
}

async function resetKnowledgeBase() {
  try {
    await apiDelete("/api/documents/reset");
    toast("Knowledge base reset", "success");
    await loadDocuments();
    await checkHealth();
  } catch (err) {
    toast(`Reset failed: ${err.message}`, "error");
  }
}

// ═══════════════════════════════════════════════════════════════════
// CHAT / SESSION
// ═══════════════════════════════════════════════════════════════════

async function createSession() {
  try {
    const data = await apiPost("/api/chat/session");
    state.sessionId = data.session_id;
    els.messages.innerHTML = "";
    els.welcomeScreen.hidden = false;
    els.headerSub.textContent = "Upload documents to start asking questions";
    return data.session_id;
  } catch (err) {
    toast(`Session error: ${err.message}`, "error");
    return null;
  }
}

async function sendMessage() {
  const text = els.chatInput.value.trim();
  if (!text || state.isStreaming) return;

  // Ensure session exists
  if (!state.sessionId) {
    const sid = await createSession();
    if (!sid) return;
  }

  // Hide welcome, show messages
  els.welcomeScreen.hidden = true;

  // Add user bubble
  appendMessage("user", text);

  // Clear input
  els.chatInput.value = "";
  els.chatInput.style.height = "auto";
  updateCharCount();
  updateSendBtn();

  // Show loading
  const loadingEl = appendLoading();

  state.isStreaming = true;
  updateSendBtn();
  els.headerSub.textContent = "Thinking...";

  try {
    const eventSource = new EventSource(
      `${API}/api/chat/${state.sessionId}/message?` +
      new URLSearchParams({ _method: "POST" })
    );
    // EventSource only supports GET. We'll use fetch + ReadableStream instead.
    await streamResponse(text, loadingEl);
  } catch (err) {
    loadingEl.remove();
    appendMessage("assistant", `⚠ Error: ${err.message}`);
    toast(`Chat error: ${err.message}`, "error");
  } finally {
    state.isStreaming = false;
    updateSendBtn();
    els.headerSub.textContent = `Session ${state.sessionId?.slice(0, 8) || "—"}`;
  }
}

async function streamResponse(text, loadingEl) {
  const res = await fetch(`${API}/api/chat/${state.sessionId}/message`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: text }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }

  // Remove loading spinner
  loadingEl.remove();

  // Create assistant bubble for streaming
  const { bubbleEl, bodyEl } = appendStreamingBubble();
  let fullText = "";
  let sources = [];

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (!payload) continue;

      try {
        const event = JSON.parse(payload);

        if (event.type === "token") {
          fullText += event.content;
          bodyEl.innerHTML = renderMarkdown(fullText);
          scrollToBottom();
        } else if (event.type === "sources") {
          sources = event.sources || [];
        } else if (event.type === "error") {
          fullText += `\n\n⚠ ${event.message}`;
          bodyEl.innerHTML = renderMarkdown(fullText);
        } else if (event.type === "done") {
          // Streaming complete
        }
      } catch (e) {
        // ignore malformed JSON
      }
    }
  }

  // Remove streaming cursor
  bubbleEl.classList.remove("streaming");

  // Add sources panel if we have any
  if (sources.length > 0) {
    const sourcesEl = buildSourcesPanel(sources);
    bubbleEl.closest(".message-body").appendChild(sourcesEl);
  }

  // Add timestamp
  const timeEl = document.createElement("span");
  timeEl.className = "message-time";
  timeEl.textContent = formatTime();
  bubbleEl.closest(".message-body").appendChild(timeEl);

  scrollToBottom();
}

// ═══════════════════════════════════════════════════════════════════
// MESSAGE DOM HELPERS
// ═══════════════════════════════════════════════════════════════════

function appendMessage(role, content) {
  const msg = document.createElement("div");
  msg.className = `message ${role}`;

  const avatarIcon = role === "user" ? "👤" : "🤖";
  msg.innerHTML = `
    <div class="message-avatar">${avatarIcon}</div>
    <div class="message-body">
      <div class="message-bubble">${role === "user" ? escapeHtml(content) : renderMarkdown(content)}</div>
      <span class="message-time">${formatTime()}</span>
    </div>
  `;
  els.messages.appendChild(msg);
  scrollToBottom();
  return msg;
}

function appendLoading() {
  const msg = document.createElement("div");
  msg.className = "message assistant message-loading";
  msg.innerHTML = `
    <div class="message-avatar">🤖</div>
    <div class="message-body">
      <div class="dot-spin"><span></span><span></span><span></span></div>
    </div>
  `;
  els.messages.appendChild(msg);
  scrollToBottom();
  return msg;
}

function appendStreamingBubble() {
  const msg = document.createElement("div");
  msg.className = "message assistant";
  msg.innerHTML = `
    <div class="message-avatar">🤖</div>
    <div class="message-body">
      <div class="message-bubble streaming"></div>
    </div>
  `;
  els.messages.appendChild(msg);
  const bubbleEl = msg.querySelector(".message-bubble");
  return { bubbleEl, bodyEl: bubbleEl };
}

function buildSourcesPanel(sources) {
  const panel = document.createElement("div");
  panel.className = "sources-panel";

  const toggle = document.createElement("button");
  toggle.className = "sources-toggle";
  toggle.innerHTML = `
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
    Sources
    <span class="sources-badge">${sources.length}</span>
  `;

  const list = document.createElement("div");
  list.className = "sources-list";

  sources.forEach((src) => {
    const item = document.createElement("div");
    item.className = "source-item";
    item.innerHTML = `
      <span class="source-num">${src.index}</span>
      <div class="source-content">
        <div class="source-file">${escapeHtml(src.filename)}</div>
        <div class="source-snippet">${escapeHtml(src.snippet)}</div>
      </div>
      <span class="source-score">${(src.relevance_score * 100).toFixed(0)}%</span>
    `;
    list.appendChild(item);
  });

  toggle.addEventListener("click", () => {
    toggle.classList.toggle("open");
    list.classList.toggle("open");
  });

  panel.appendChild(toggle);
  panel.appendChild(list);
  return panel;
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    els.chatArea.scrollTop = els.chatArea.scrollHeight;
  });
}

// ═══════════════════════════════════════════════════════════════════
// CONFIRMATION MODAL
// ═══════════════════════════════════════════════════════════════════

let _modalResolve = null;

function confirmAction(title, body, onConfirm) {
  els.modalTitle.textContent = title;
  els.modalBody.textContent = body;
  els.confirmModal.hidden = false;

  const cleanup = () => {
    els.confirmModal.hidden = true;
    els.modalConfirm.removeEventListener("click", handleConfirm);
    els.modalCancel.removeEventListener("click", handleCancel);
  };

  const handleConfirm = () => { cleanup(); onConfirm(); };
  const handleCancel = () => { cleanup(); };

  els.modalConfirm.addEventListener("click", handleConfirm);
  els.modalCancel.addEventListener("click", handleCancel);
}

// ═══════════════════════════════════════════════════════════════════
// INPUT HANDLING
// ═══════════════════════════════════════════════════════════════════

function updateCharCount() {
  const len = els.chatInput.value.length;
  els.charCount.textContent = `${len} / 4000`;
}

function updateSendBtn() {
  els.sendBtn.disabled = !els.chatInput.value.trim() || state.isStreaming;
}

function autoResize() {
  els.chatInput.style.height = "auto";
  els.chatInput.style.height = Math.min(els.chatInput.scrollHeight, 180) + "px";
}

// ═══════════════════════════════════════════════════════════════════
// EVENT LISTENERS
// ═══════════════════════════════════════════════════════════════════

function initEvents() {
  // ── Sidebar toggle ────────────────────────────────────────────
  els.sidebarToggle.addEventListener("click", () => {
    els.sidebar.classList.toggle("collapsed");
  });
  els.mobileMenuBtn.addEventListener("click", () => {
    els.sidebar.classList.toggle("mobile-open");
  });
  // Close mobile sidebar when clicking main area
  document.getElementById("main").addEventListener("click", () => {
    if (window.innerWidth <= 768) {
      els.sidebar.classList.remove("mobile-open");
    }
  });

  // ── New chat ──────────────────────────────────────────────────
  els.newChatBtn.addEventListener("click", async () => {
    await createSession();
    els.messages.innerHTML = "";
    els.welcomeScreen.hidden = false;
    toast("New conversation started", "info");
  });

  // ── Clear chat ────────────────────────────────────────────────
  els.clearChatBtn.addEventListener("click", () => {
    if (!state.sessionId) return;
    confirmAction("Clear Conversation", "Delete this conversation history?", async () => {
      try {
        await apiDelete(`/api/chat/${state.sessionId}`);
        await createSession();
        els.messages.innerHTML = "";
        els.welcomeScreen.hidden = false;
        toast("Conversation cleared", "info");
      } catch (err) {
        toast(`Clear failed: ${err.message}`, "error");
      }
    });
  });

  // ── File upload ───────────────────────────────────────────────
  els.uploadZone.addEventListener("click", () => els.fileInput.click());
  els.fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) uploadFiles(e.target.files);
    e.target.value = "";
  });

  // Drag & drop
  els.uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    els.uploadZone.classList.add("drag-over");
  });
  els.uploadZone.addEventListener("dragleave", () => {
    els.uploadZone.classList.remove("drag-over");
  });
  els.uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    els.uploadZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
  });

  // ── Refresh docs ──────────────────────────────────────────────
  els.refreshDocsBtn.addEventListener("click", loadDocuments);

  // ── Reset KB ──────────────────────────────────────────────────
  els.resetKbBtn.addEventListener("click", () => {
    confirmAction(
      "Reset Knowledge Base",
      "This will permanently delete ALL uploaded documents. This action cannot be undone.",
      resetKnowledgeBase
    );
  });

  // ── Chat input ────────────────────────────────────────────────
  els.chatInput.addEventListener("input", () => {
    updateCharCount();
    updateSendBtn();
    autoResize();
  });

  els.chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  els.sendBtn.addEventListener("click", sendMessage);

  // ── Suggestion chips ──────────────────────────────────────────
  els.suggestionChips.addEventListener("click", (e) => {
    if (e.target.classList.contains("chip")) {
      els.chatInput.value = e.target.textContent;
      updateCharCount();
      updateSendBtn();
      autoResize();
      els.chatInput.focus();
    }
  });

  // ── Modal overlay click ───────────────────────────────────────
  els.confirmModal.addEventListener("click", (e) => {
    if (e.target === els.confirmModal) {
      els.confirmModal.hidden = true;
    }
  });
}

// ═══════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════

async function init() {
  initEvents();
  await checkHealth();
  await loadDocuments();
  await createSession();

  // Periodic health check every 30s
  setInterval(checkHealth, 30000);

  console.log("RAG AI Agent — Frontend initialised ✓");
}

document.addEventListener("DOMContentLoaded", init);
