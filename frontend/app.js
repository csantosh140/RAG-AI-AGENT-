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
  voiceMode: false,
  webSearchEnabled: true,
  attachedFiles: [],   // Files attached inline in the search bar
};

// ── DOM Elements ───────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const els = {
  sidebar:        $("#sidebar"),
  sidebarToggle:  $("#sidebarToggle"),
  mobileMenuBtn:  $("#mobileMenuBtn"),
  statusDot:      $("#statusDot"),
  statusText:     $("#statusText"),
  statusDocs:     $("#statusDocs"),
  newChatBtn:     $("#newChatBtn"),
  refreshDocsBtn: $("#refreshDocsBtn"),
  uploadZone:     $("#uploadZone"),
  fileInput:      $("#fileInput"),
  uploadProgress: $("#uploadProgress"),
  progressFill:   $("#progressFill"),
  progressLabel:  $("#progressLabel"),
  docList:        $("#docList"),
  docEmpty:       $("#docEmpty"),
  chatList:       $("#chatList"),
  chatEmpty:      $("#chatEmpty"),
  resetKbBtn:     $("#resetKbBtn"),
  chatArea:       $("#chatArea"),
  welcomeScreen:  $("#welcomeScreen"),
  messages:       $("#messages"),
  chatInput:      $("#chatInput"),
  charCount:      null,
  sendBtn:        $("#sendBtn"),
  voiceBtn:       $("#voiceBtn"),
  attachBtn:      $("#attachBtn"),
  chatFileInput:  $("#chatFileInput"),
  attachedFilesPreview: $("#attachedFilesPreview"),
  generateQuizBtn:$("#generateQuizBtn"),
  generatePdfBtn: $("#generatePdfBtn"),
  pdfTopicModal:   $("#pdfTopicModal"),
  pdfTopicInput:   $("#pdfTopicInput"),
  pdfTopicCancel:  $("#pdfTopicCancel"),
  pdfTopicConfirm: $("#pdfTopicConfirm"),
  clearChatBtn:   $("#clearChatBtn"),
  headerTitle:    $("#headerTitle"),
  headerSub:      $("#headerSub"),
  toastContainer: $("#toastContainer"),
  confirmModal:   $("#confirmModal"),
  modalTitle:     $("#modalTitle"),
  modalBody:      $("#modalBody"),
  modalCancel:    $("#modalCancel"),
  modalConfirm:   $("#modalConfirm"),
  suggestionChips:$("#suggestionChips"),
  topicSearchBtn:   $("#topicSearchBtn"),
  topicSearchOverlay: $("#topicSearchOverlay"),
  topicSearchPanel:  $("#topicSearchPanel"),
  topicSearchClose:  $("#topicSearchClose"),
  topicSearchInput:  $("#topicSearchInput"),
  topicSearchGo:     $("#topicSearchGo"),
  topicSearchStats:  $("#topicSearchStats"),
  topicSearchResults:$("#topicSearchResults"),
  topicEmptyState:   $("#topicEmptyState"),
  topicLoading:      $("#topicLoading"),
  webSearchToggle:  $("#webSearchToggle"),
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
    png: "🖼️", jpg: "🖼️", jpeg: "🖼️", gif: "🎞️", webp: "🖼️", bmp: "🖼️", svg: "🎨",
  };
  return { icon: icons[ext] || "📄", ext };
}

const IMAGE_EXTENSIONS = new Set(['png','jpg','jpeg','gif','webp','bmp','svg']);
function isImageFile(filename) {
  return IMAGE_EXTENSIONS.has(filename.split('.').pop().toLowerCase());
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function getAttachChipClass(filename) {
  const ext = filename.split('.').pop().toLowerCase();
  if (IMAGE_EXTENSIONS.has(ext)) return 'image';
  if (['pdf'].includes(ext)) return 'pdf';
  if (['docx','doc'].includes(ext)) return 'docx';
  if (['txt','md'].includes(ext)) return 'txt';
  if (['csv'].includes(ext)) return 'csv';
  if (['html','htm'].includes(ext)) return 'html';
  return 'other';
}

/** Chart type keywords to detect in JSON blocks */
const CHART_TYPES = ['bar_chart', 'line_chart', 'pie_chart', 'scatter_plot', 'histogram', 'comparison', 'infographic', 'timeline', 'ai_photo'];
const CHART_TYPE_REGEX = new RegExp(`"type"\\s*:\\s*"(${CHART_TYPES.join('|')})"`, 'i');

/** Clean chart JSON blocks from text (all formats) */
function cleanChartBlocks(text) {
  if (!text) return text;

  const placeholder = '\n\n📊 *Generating visualization...*\n\n';

  // 1. ```chart_json ... ```
  text = text.replace(/```chart_json[\s\S]*?```/g, placeholder);

  // 2. ```json ... ``` containing a chart type
  text = text.replace(/```json\s*\n?([\s\S]*?)```/g, (match, content) => {
    return CHART_TYPE_REGEX.test(content) ? placeholder : match;
  });

  // 3. ``` ... ``` generic code blocks containing chart JSON
  text = text.replace(/```\s*\n?(\{[\s\S]*?"type"\s*:[\s\S]*?\})\s*\n?```/g, (match, content) => {
    return CHART_TYPE_REGEX.test(content) ? placeholder : match;
  });

  // 4. Raw JSON blocks (multi-line) matching chart types
  for (const ct of CHART_TYPES) {
    // Multi-line: { "type": "bar_chart", ... }
    const rawMulti = new RegExp(`\\{\\s*\\n[\\s\\S]*?"type"\\s*:\\s*"${ct}"[\\s\\S]*?\\n\\}`, 'g');
    text = text.replace(rawMulti, placeholder);
    // Single-line
    const rawSingle = new RegExp(`\\{[^{}]*"type"\\s*:\\s*"${ct}"[^{}]*\\}`, 'g');
    text = text.replace(rawSingle, placeholder);
  }

  // 5. Hide partial/incomplete JSON blocks that are still streaming
  //    e.g., ```chart_json\n{... (no closing ```)
  text = text.replace(/```chart_json[\s\S]*$/g, '\n\n⏳ *Generating chart...*');
  // Partial ```json{ with chart type
  text = text.replace(/```json\s*\n?\{[\s\S]*$/g, (match) => {
    return CHART_TYPE_REGEX.test(match) ? '\n\n⏳ *Generating chart...*' : match;
  });

  return text;
}

/** Minimal markdown → HTML.  Handles bold, italic, code, headings, lists, blockquotes, links */
function renderMarkdown(text) {
  if (!text) return "";
  
  // First, clean all chart JSON blocks from the text
  text = cleanChartBlocks(text);
  
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

function speakText(markdownText) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel(); // stop any ongoing speech
  
  // Remove markdown symbols and markdown links
  let plain = markdownText
    .replace(/\*\*/g, "")
    .replace(/\*/g, "")
    .replace(/#/g, "")
    .replace(/`/g, "")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1") // Links
    .replace(/\[\d+\]/g, ""); // Source citations [1]
    
  const utterance = new SpeechSynthesisUtterance(plain);
  
  // Optional: Set voice language based on text if detected or let OS default
  window.speechSynthesis.speak(utterance);
}


// ═══════════════════════════════════════════════════════════════════
// API CALLS
// ═══════════════════════════════════════════════════════════════════

async function apiGet(path) {
  const headers = {};
  const token = localStorage.getItem("token");
  if (token) headers["Authorization"] = `Bearer ${token}`;
  
  const res = await fetch(`${API}${path}`, { headers });
  if (res.status === 401 && !path.includes("/health")) {
    showAuthModal();
    throw new Error("Unauthorized");
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}


async function apiPost(path, body, isJson = true) {
  const opts = { method: "POST" };
  opts.headers = {};
  const token = localStorage.getItem("token");
  if (token) opts.headers["Authorization"] = `Bearer ${token}`;

  if (isJson) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  } else {
    opts.body = body; // FormData
  }
  const res = await fetch(`${API}${path}`, opts);
  if (res.status === 401) {
    showAuthModal();
    throw new Error("Unauthorized");
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}


async function apiDelete(path) {
  const headers = {};
  const token = localStorage.getItem("token");
  if (token) headers["Authorization"] = `Bearer ${token}`;
  
  const res = await fetch(`${API}${path}`, { method: "DELETE", headers });
  if (res.status === 401) {
    showAuthModal();
    throw new Error("Unauthorized");
  }
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
    if (window.innerWidth <= 768) {
      els.sidebar.classList.remove("mobile-open");
    }
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
    if (window.innerWidth <= 768) {
      els.sidebar.classList.remove("mobile-open");
    }
  } catch (err) {
    toast(`Reset failed: ${err.message}`, "error");
  }
}

// ═══════════════════════════════════════════════════════════════════
// CHAT / SESSION
// ═══════════════════════════════════════════════════════════════════

async function loadChatHistory() {
  try {
    const history = await apiGet("/api/chat/history");
    renderChatHistory(history);
  } catch (err) {
    console.error("Failed to load chat history:", err);
  }
}

function renderChatHistory(history) {
  // Clear existing items
  els.chatList.querySelectorAll(".chat-item").forEach((el) => el.remove());

  // Filter out sessions that have no messages
  const activeSessions = (history || []).filter(s => s.messages_count > 0);

  if (activeSessions.length === 0) {
    els.chatEmpty.hidden = false;
    return;
  }
  els.chatEmpty.hidden = true;

  activeSessions.forEach((session) => {
    const item = document.createElement("div");
    item.className = "chat-item";
    item.dataset.id = session.session_id;
    if (state.sessionId === session.session_id) {
      item.classList.add("active");
    }
    item.innerHTML = `
      <div class="chat-icon">💬</div>
      <div class="chat-info">
        <div class="chat-name" title="${escapeHtml(session.title)}">${escapeHtml(session.title)}</div>
        <div class="chat-meta">${session.messages_count} messages</div>
      </div>
      <button class="chat-delete" data-id="${session.session_id}" title="Delete conversation">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
      </button>
    `;
    item.addEventListener("click", () => {
      loadChatSession(session.session_id);
      if (window.innerWidth <= 768) {
        els.sidebar.classList.remove("mobile-open");
      }
    });
    item.querySelector(".chat-delete").addEventListener("click", (e) => {
      e.stopPropagation();
      confirmAction(
        "Delete Conversation",
        `Remove this conversation from history? This cannot be undone.`,
        () => deleteChatSession(session.session_id)
      );
    });
    els.chatList.appendChild(item);
  });
}

function updateActiveSessionInSidebar(sessionId) {
  const items = els.chatList.querySelectorAll(".chat-item");
  items.forEach((item) => {
    if (item.dataset.id === sessionId) {
      item.classList.add("active");
    } else {
      item.classList.remove("active");
    }
  });
}

async function deleteChatSession(sessionId) {
  try {
    await apiDelete(`/api/chat/${sessionId}`);
    toast("Conversation deleted", "success");
    if (state.sessionId === sessionId) {
      await createSession();
      els.messages.innerHTML = "";
      els.welcomeScreen.hidden = false;
    }
    await loadChatHistory();
  } catch (err) {
    toast(`Delete failed: ${err.message}`, "error");
  }
}

async function loadChatSession(sessionId) {
  if (state.isStreaming) return;
  try {
    const data = await apiGet(`/api/chat/session/${sessionId}`);
    state.sessionId = sessionId;
    
    // Clear chat area
    els.messages.innerHTML = "";
    
    if (data.messages && data.messages.length > 0) {
      els.welcomeScreen.hidden = true;
      
      data.messages.forEach(msg => {
        const avatarIcon = msg.role === "user" ? "👤" : "🤖";
        
        const messageDiv = document.createElement("div");
        messageDiv.className = `message ${msg.role}`;
        
        messageDiv.innerHTML = `
          <div class="message-avatar">${avatarIcon}</div>
          <div class="message-body">
            <div class="message-bubble">${msg.role === "user" ? escapeHtml(msg.content) : renderMarkdown(msg.content)}</div>
          </div>
        `;
        
        const bodyContainer = messageDiv.querySelector(".message-body");
        
        // If assistant message has search_sources, render them:
        if (msg.role === "assistant" && msg.search_sources) {
          const sourcesObj = msg.search_sources;
          
          // Render generated images if any
          const generatedImages = sourcesObj.generated_images || [];
          if (generatedImages.length > 0) {
            const imagesContainer = document.createElement("div");
            imagesContainer.className = "generated-images-container";
            
            generatedImages.forEach((img, idx) => {
              const imageCard = document.createElement("div");
              imageCard.className = "generated-image-card";
              imageCard.style.animationDelay = `${idx * 0.15}s`;
              imageCard.innerHTML = `
                <div class="gen-image-header">
                  <span class="gen-image-badge">${img.chart_type === 'ai_photo' ? '🎨 AI Generated Image' : '📊 ' + escapeHtml(img.chart_type.replace('_', ' '))}</span>
                  <span class="gen-image-title">${escapeHtml(img.title)}</span>
                  <button class="gen-image-download" title="Download Image" onclick="downloadGenImage('${img.image_id}', '${escapeHtml(img.title)}')">  
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                      <polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
                    </svg>
                  </button>
                </div>
                <div class="gen-image-wrap">
                  <img src="${API}/api/images/${img.image_id}" alt="${escapeHtml(img.title)}" class="gen-image" loading="lazy" />
                </div>
              `;
              imagesContainer.appendChild(imageCard);
            });
            bodyContainer.appendChild(imagesContainer);
          }
          
          // Render document sources if any
          const docSources = sourcesObj.doc_sources || [];
          if (docSources.length > 0) {
            const sourcesEl = buildSourcesPanel(docSources);
            bodyContainer.appendChild(sourcesEl);
          }
          
          // Render web sources if any
          const webSources = sourcesObj.web_sources || [];
          if (webSources.length > 0) {
            const webSourcesEl = buildWebSourcesPanel(webSources);
            bodyContainer.appendChild(webSourcesEl);
          }
        }
        
        // Add timestamp
        const timeEl = document.createElement("span");
        timeEl.className = "message-time";
        timeEl.textContent = formatTime(msg.created_at);
        bodyContainer.appendChild(timeEl);
        
        els.messages.appendChild(messageDiv);
      });
      
      scrollToBottom();
    } else {
      els.welcomeScreen.hidden = false;
    }
    
    // Update subheader
    els.headerSub.textContent = `Session ${sessionId.slice(0, 8)}`;
    
    // Highlight selected chat item in sidebar
    updateActiveSessionInSidebar(sessionId);
    
  } catch (err) {
    toast(`Failed to load conversation: ${err.message}`, "error");
  }
}

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
  const hasFiles = state.attachedFiles.length > 0;
  if ((!text && !hasFiles) || state.isStreaming) return;

  // Ensure session exists
  if (!state.sessionId) {
    const sid = await createSession();
    if (!sid) return;
  }

  // Hide welcome, show messages
  els.welcomeScreen.hidden = true;

  // Capture attached file names before clearing
  const attachedFileNames = state.attachedFiles.map(f => f.name);
  const attachedFileObjects = [...state.attachedFiles];

  // Add user bubble with attachment badges
  appendMessage("user", text || "(attached files)", attachedFileNames);

  // Clear input & attachments
  els.chatInput.value = "";
  clearAttachedFiles();
  updateCharCount();
  updateSendBtn();

  // Show loading
  const loadingEl = appendLoading();

  state.isStreaming = true;
  updateSendBtn();
  els.headerSub.textContent = hasFiles ? "Uploading files & thinking..." : "Thinking...";

  try {
    // If files are attached, upload them to the knowledge base first
    if (attachedFileObjects.length > 0) {
      els.headerSub.textContent = "Uploading files...";
      await uploadFiles(attachedFileObjects);
      els.headerSub.textContent = "Thinking...";
    }

    let queryText = text || `Please analyze and summarize the content of the attached files.`;
    if (attachedFileNames.length > 0) {
      queryText = `[AttachedFiles: ${attachedFileNames.join(',')}] ${queryText}`;
    }
    await streamResponse(queryText, loadingEl);
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
  const webParam = state.webSearchEnabled ? 'true' : 'false';
  const token = localStorage.getItem("token");
  const headers = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  
  const res = await fetch(`${API}/api/chat/${state.sessionId}/message?web_search=${webParam}`, {
    method: "POST",
    headers: headers,
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
  let webSources = [];
  let generatedImages = [];
  let statusShown = false;

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
        } else if (event.type === "web_sources") {
          webSources = event.sources || [];
        } else if (event.type === "image") {
          // AI generated a chart/graph!
          generatedImages.push(event);
        } else if (event.type === "status") {
          // Show status message (e.g., "Searching the web...", "Generating chart...")
          if (!statusShown) {
            const statusEl = document.createElement("div");
            statusEl.className = "llm-status-msg";
            statusEl.innerHTML = `<span class="llm-status-dot"></span> ${escapeHtml(event.message)}`;
            bubbleEl.closest(".message-body").insertBefore(statusEl, bubbleEl);
            statusShown = true;
            // Auto-remove after content starts flowing
            setTimeout(() => { statusEl.style.opacity = '0'; setTimeout(() => statusEl.remove(), 300); }, 3000);
          } else {
            // Update existing status message for subsequent statuses
            const existingStatus = bubbleEl.closest(".message-body").querySelector('.llm-status-msg');
            if (existingStatus) {
              existingStatus.innerHTML = `<span class="llm-status-dot"></span> ${escapeHtml(event.message)}`;
              existingStatus.style.opacity = '1';
              setTimeout(() => { existingStatus.style.opacity = '0'; setTimeout(() => existingStatus.remove(), 300); }, 3000);
            }
          }
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

  // Render generated images/charts inline
  if (generatedImages.length > 0) {
    const imagesContainer = document.createElement("div");
    imagesContainer.className = "generated-images-container";
    
    generatedImages.forEach((img, idx) => {
      const imageCard = document.createElement("div");
      imageCard.className = "generated-image-card";
      imageCard.style.animationDelay = `${idx * 0.15}s`;
      imageCard.innerHTML = `
        <div class="gen-image-header">
          <span class="gen-image-badge">${img.chart_type === 'ai_photo' ? '🎨 AI Generated Image' : '📊 ' + escapeHtml(img.chart_type.replace('_', ' '))}</span>
          <span class="gen-image-title">${escapeHtml(img.title)}</span>
          <button class="gen-image-download" title="Download Image" onclick="downloadGenImage('${img.image_id}', '${escapeHtml(img.title)}')">  
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
          </button>
        </div>
        <div class="gen-image-wrap">
          <img src="${API}/api/images/${img.image_id}" alt="${escapeHtml(img.title)}" class="gen-image" loading="lazy" />
        </div>
      `;
      imagesContainer.appendChild(imageCard);
    });
    
    bubbleEl.closest(".message-body").appendChild(imagesContainer);
  }

  // Add sources panel if we have any
  if (sources.length > 0) {
    const sourcesEl = buildSourcesPanel(sources);
    bubbleEl.closest(".message-body").appendChild(sourcesEl);
  }

  // Add web sources panel if we have any
  if (webSources.length > 0) {
    const webSourcesEl = buildWebSourcesPanel(webSources);
    bubbleEl.closest(".message-body").appendChild(webSourcesEl);
  }

  // Add timestamp
  const timeEl = document.createElement("span");
  timeEl.className = "message-time";
  timeEl.textContent = formatTime();
  bubbleEl.closest(".message-body").appendChild(timeEl);

  scrollToBottom();
  
  // Reload chat history to update messages count and titles
  await loadChatHistory();
  
  // Text-to-Speech if voice mode was used
  if (state.voiceMode && fullText.trim()) {
    speakText(fullText);
    state.voiceMode = false; // Reset for next message
  }
}

// Download generated image
function downloadGenImage(imageId, title) {
  const link = document.createElement('a');
  link.href = `${API}/api/images/${imageId}`;
  link.download = `${title.replace(/[^a-zA-Z0-9]/g, '_')}.png`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  toast('Image downloaded!', 'success');
}

// Fullscreen image viewer
document.addEventListener('click', (e) => {
  const img = e.target.closest('.gen-image');
  if (!img) return;
  
  const overlay = document.createElement('div');
  overlay.className = 'gen-image-fullscreen-overlay';
  overlay.innerHTML = `<img src="${img.src}" alt="${img.alt}" />`;
  overlay.addEventListener('click', () => {
    overlay.style.opacity = '0';
    setTimeout(() => overlay.remove(), 200);
  });
  document.body.appendChild(overlay);
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const overlay = document.querySelector('.gen-image-fullscreen-overlay');
    if (overlay) {
      overlay.style.opacity = '0';
      setTimeout(() => overlay.remove(), 200);
    }
  }
});

// ═══════════════════════════════════════════════════════════════════
// MESSAGE DOM HELPERS
// ═══════════════════════════════════════════════════════════════════

function appendMessage(role, content, attachedFileNames = []) {
  const msg = document.createElement("div");
  msg.className = `message ${role}`;

  const avatarIcon = role === "user" ? "👤" : "🤖";
  
  // Build attachment badges HTML if there are attached files
  let attachmentsHtml = '';
  if (attachedFileNames.length > 0 && role === 'user') {
    attachmentsHtml = `<div class="msg-attachments">`;
    attachedFileNames.forEach(fname => {
      const { icon } = getFileIcon(fname);
      attachmentsHtml += `<span class="msg-attach-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg> ${icon} ${escapeHtml(fname)}</span>`;
    });
    attachmentsHtml += `</div>`;
  }

  msg.innerHTML = `
    <div class="message-avatar">${avatarIcon}</div>
    <div class="message-body">
      <div class="message-bubble">${attachmentsHtml}${role === "user" ? escapeHtml(content) : renderMarkdown(content)}</div>
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
        <div class="source-file">${escapeHtml(src.filename)} (Page ${src.page || 1})</div>
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

function buildWebSourcesPanel(webSources) {
  const panel = document.createElement("div");
  panel.className = "sources-panel web-sources-panel";

  const toggle = document.createElement("button");
  toggle.className = "sources-toggle web-toggle";
  toggle.innerHTML = `
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
    🌐 Web Sources
    <span class="sources-badge web-badge">${webSources.length}</span>
  `;

  const list = document.createElement("div");
  list.className = "sources-list";

  webSources.forEach((src) => {
    const item = document.createElement("div");
    item.className = "source-item web-source-item";
    item.innerHTML = `
      <span class="source-num web-num">${src.index}</span>
      <div class="source-content">
        <div class="source-file web-source-title">${escapeHtml(src.title)}</div>
        <a href="${escapeHtml(src.url)}" target="_blank" rel="noopener" class="web-source-url">${escapeHtml(src.url.length > 60 ? src.url.slice(0, 60) + '...' : src.url)}</a>
        <div class="source-snippet">${escapeHtml(src.snippet)}</div>
      </div>
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
  if (!els.charCount) return;
  const len = els.chatInput.value.length;
  els.charCount.textContent = `${len} / 4000`;
}

function updateSendBtn() {
  const hasContent = els.chatInput.value.trim() || state.attachedFiles.length > 0;
  els.sendBtn.disabled = !hasContent || state.isStreaming;
}

function autoResize() {
  // no-op for single-line input
}

// ═══════════════════════════════════════════════════════════════════
// INLINE FILE ATTACHMENTS (Search Bar)
// ═══════════════════════════════════════════════════════════════════

const MAX_ATTACHED_FILES = 10;
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
const ALLOWED_EXTENSIONS = new Set([
  'pdf', 'docx', 'doc', 'txt', 'md', 'csv', 'html', 'htm',
  'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'svg'
]);

function addAttachedFiles(fileList) {
  const files = Array.from(fileList);
  let added = 0;

  for (const file of files) {
    // Check limits
    if (state.attachedFiles.length >= MAX_ATTACHED_FILES) {
      toast(`Maximum ${MAX_ATTACHED_FILES} files can be attached at once`, "error");
      break;
    }

    // Validate extension
    const ext = file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_EXTENSIONS.has(ext)) {
      toast(`Unsupported file type: .${ext}`, "error");
      continue;
    }

    // Validate size
    if (file.size > MAX_FILE_SIZE) {
      toast(`File too large: ${file.name} (max 50MB)`, "error");
      continue;
    }

    // Check for duplicates
    if (state.attachedFiles.some(f => f.name === file.name && f.size === file.size)) {
      toast(`Already attached: ${file.name}`, "info");
      continue;
    }

    state.attachedFiles.push(file);
    added++;
  }

  if (added > 0) {
    renderAttachedFiles();
    updateSendBtn();
    toast(`${added} file${added > 1 ? 's' : ''} attached`, "success");
  }
}

function removeAttachedFile(index) {
  state.attachedFiles.splice(index, 1);
  renderAttachedFiles();
  updateSendBtn();
}

function clearAttachedFiles() {
  state.attachedFiles = [];
  renderAttachedFiles();
}

function renderAttachedFiles() {
  const container = els.attachedFilesPreview;
  if (!container) return;

  container.innerHTML = '';

  if (state.attachedFiles.length === 0) {
    container.hidden = true;
    if (els.attachBtn) {
      els.attachBtn.classList.remove('has-files');
      els.attachBtn.removeAttribute('data-count');
    }
    return;
  }

  container.hidden = false;
  if (els.attachBtn) {
    els.attachBtn.classList.add('has-files');
    els.attachBtn.setAttribute('data-count', state.attachedFiles.length);
  }

  state.attachedFiles.forEach((file, idx) => {
    const chip = document.createElement('div');
    chip.className = 'attach-chip';

    const isImage = isImageFile(file.name);
    const { icon } = getFileIcon(file.name);
    const chipClass = getAttachChipClass(file.name);

    let thumbHtml;
    if (isImage) {
      // Create image thumbnail
      const thumbUrl = URL.createObjectURL(file);
      thumbHtml = `<img src="${thumbUrl}" class="attach-chip-thumb" alt="${escapeHtml(file.name)}" onload="URL.revokeObjectURL(this.src)" />`;
    } else {
      thumbHtml = `<div class="attach-chip-icon ${chipClass}">${icon}</div>`;
    }

    chip.innerHTML = `
      ${thumbHtml}
      <div class="attach-chip-info">
        <span class="attach-chip-name" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</span>
        <span class="attach-chip-size">${formatFileSize(file.size)}</span>
      </div>
      <button class="attach-chip-remove" title="Remove" data-idx="${idx}">
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
    `;

    chip.querySelector('.attach-chip-remove').addEventListener('click', (e) => {
      e.stopPropagation();
      removeAttachedFile(idx);
    });

    container.appendChild(chip);
  });
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

  // ── Header Actions Dropdown (Mobile) ──────────────────────────
  const moreBtn = document.getElementById("headerMoreBtn");
  const dropdown = document.getElementById("headerDropdown");
  if (moreBtn && dropdown) {
    moreBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropdown.hidden = !dropdown.hidden;
    });

    document.addEventListener("click", () => {
      dropdown.hidden = true;
    });

    dropdown.addEventListener("click", (e) => {
      e.stopPropagation();
    });

    document.getElementById("dropdownWebSearch")?.addEventListener("click", () => {
      document.getElementById("webSearchToggle")?.click();
      dropdown.hidden = true;
    });

    document.getElementById("dropdownQuiz")?.addEventListener("click", () => {
      document.getElementById("generateQuizBtn")?.click();
      dropdown.hidden = true;
    });

    document.getElementById("dropdownPdf")?.addEventListener("click", () => {
      document.getElementById("generatePdfBtn")?.click();
      dropdown.hidden = true;
    });

    document.getElementById("dropdownClear")?.addEventListener("click", () => {
      document.getElementById("clearChatBtn")?.click();
      dropdown.hidden = true;
    });
  }

  // ── Web Search Toggle ─────────────────────────────────────────
  if (els.webSearchToggle) {
    els.webSearchToggle.addEventListener("click", () => {
      state.webSearchEnabled = !state.webSearchEnabled;
      els.webSearchToggle.classList.toggle("active", state.webSearchEnabled);
      els.webSearchToggle.title = state.webSearchEnabled
        ? "Web Search: ON — Click to toggle"
        : "Web Search: OFF — Click to toggle";
      
      const dropWeb = document.getElementById("dropdownWebSearch");
      if (dropWeb) {
        dropWeb.classList.toggle("active", state.webSearchEnabled);
      }
      
      toast(
        state.webSearchEnabled
          ? "🌐 Web search enabled — AI will augment answers with live web data"
          : "📄 Web search disabled — AI will use only your uploaded documents",
        "info"
      );
    });
  }

  // ── New chat ──────────────────────────────────────────────────
  els.newChatBtn.addEventListener("click", async () => {
    await createSession();
    els.messages.innerHTML = "";
    els.welcomeScreen.hidden = false;
    toast("New conversation started", "info");
    await loadChatHistory();
    if (window.innerWidth <= 768) {
      els.sidebar.classList.remove("mobile-open");
    }
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
        await loadChatHistory();
      } catch (err) {
        toast(`Clear failed: ${err.message}`, "error");
      }
    });
  });

  // ── Generate Quiz ─────────────────────────────────────────────
  if (els.generateQuizBtn) {
    els.generateQuizBtn.addEventListener("click", async () => {
      if (state.isStreaming) return;
      if (state.documents.length === 0) {
        toast("Please upload a document first to generate a quiz.", "warn");
        return;
      }
      
      els.welcomeScreen.hidden = true;
      appendMessage("user", "Please generate a quiz based on the uploaded documents.");
      
      const loadingEl = appendLoading();
      state.isStreaming = true;
      updateSendBtn();
      els.headerSub.textContent = "Generating Quiz...";
      
      try {
        const res = await fetch(`${API}/api/quiz/generate`, { method: "GET" });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || res.statusText);
        }
        
        loadingEl.remove();
        const { bubbleEl, bodyEl } = appendStreamingBubble();
        let fullText = "";
        
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
              } else if (event.type === "error") {
                fullText += `\n\n⚠ ${event.message}`;
                bodyEl.innerHTML = renderMarkdown(fullText);
              }
            } catch (e) {}
          }
        }
        bubbleEl.classList.remove("streaming");
        
        const timeEl = document.createElement("span");
        timeEl.className = "message-time";
        timeEl.textContent = formatTime();
        bubbleEl.closest(".message-body").appendChild(timeEl);
        scrollToBottom();
        
      } catch (err) {
        loadingEl.remove();
        appendMessage("assistant", `⚠ Error: ${err.message}`);
        toast(`Quiz generation error: ${err.message}`, "error");
      } finally {
        state.isStreaming = false;
        updateSendBtn();
        els.headerSub.textContent = `Session ${state.sessionId?.slice(0, 8) || "—"}`;
      }
    });
  }

  // ── Generate PDF Report ───────────────────────────────────────────
  if (els.generatePdfBtn) {
    els.generatePdfBtn.addEventListener("click", () => {
      els.pdfTopicInput.value = "";
      els.pdfTopicModal.hidden = false;
      setTimeout(() => els.pdfTopicInput.focus(), 100);
    });
  }

  if (els.pdfTopicCancel) {
    els.pdfTopicCancel.addEventListener("click", () => {
      els.pdfTopicModal.hidden = true;
    });
  }

  if (els.pdfTopicModal) {
    els.pdfTopicModal.addEventListener("click", (e) => {
      if (e.target === els.pdfTopicModal) {
        els.pdfTopicModal.hidden = true;
      }
    });
  }

  if (els.pdfTopicInput) {
    els.pdfTopicInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        els.pdfTopicConfirm.click();
      }
    });
  }

  if (els.pdfTopicConfirm) {
    els.pdfTopicConfirm.addEventListener("click", async () => {
      const topic = els.pdfTopicInput.value.trim();
      if (!topic) {
        toast("Please enter a topic", "warn");
        return;
      }

      els.pdfTopicModal.hidden = true;
      toast(`Generating PDF report on "${topic}"...`, "info");
      els.headerSub.textContent = "Generating PDF Document...";
      
      try {
        const response = await fetch(`${API}/api/documents/generate-pdf?topic=${encodeURIComponent(topic)}`, {
          method: "POST"
        });

        if (!response.ok) {
          const err = await response.json().catch(() => ({}));
          throw new Error(err.detail || response.statusText);
        }

        const filename = response.headers.get("X-Document-Filename") || `Report_${topic.replace(/\s+/g, '_')}.pdf`;
        
        const blob = await response.blob();
        const downloadUrl = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = downloadUrl;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(downloadUrl);

        toast(`PDF report "${filename}" generated & indexed successfully!`, "success");
        await loadDocuments();
        await checkHealth();
      } catch (err) {
        toast(`PDF generation failed: ${err.message}`, "error");
      } finally {
        els.headerSub.textContent = `Session ${state.sessionId?.slice(0, 8) || "—"}`;
      }
    });
  }

  // ── File upload (sidebar) ─────────────────────────────────────
  els.uploadZone.addEventListener("click", () => els.fileInput.click());
  els.fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) uploadFiles(e.target.files);
    e.target.value = "";
  });

  // Drag & drop (sidebar)
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

  // ── Inline file attach (search bar) ──────────────────────────
  const attachBtn = document.getElementById('attachBtn');
  const chatFileInput = document.getElementById('chatFileInput');

  if (attachBtn && chatFileInput) {
    attachBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      chatFileInput.value = "";  // Reset so same file can be re-selected
      chatFileInput.click();
    });

    chatFileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        addAttachedFiles(e.target.files);
      }
    });
  } else {
    console.warn("Attach button or file input not found:", { attachBtn, chatFileInput });
  }

  // Drag & drop on search bar
  const searchBar = document.getElementById("searchBar");
  if (searchBar) {
    searchBar.addEventListener("dragover", (e) => {
      e.preventDefault();
      searchBar.classList.add("drag-active");
    });
    searchBar.addEventListener("dragleave", (e) => {
      if (!searchBar.contains(e.relatedTarget)) {
        searchBar.classList.remove("drag-active");
      }
    });
    searchBar.addEventListener("drop", (e) => {
      e.preventDefault();
      searchBar.classList.remove("drag-active");
      if (e.dataTransfer.files.length) addAttachedFiles(e.dataTransfer.files);
    });
  }

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

  // ── Voice Assistant ───────────────────────────────────────────
  if (els.voiceBtn) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      // recognition.lang = 'en-US'; // Leave default for auto-detection

      recognition.onstart = () => {
        els.voiceBtn.classList.add("listening");
        els.chatInput.placeholder = "Listening...";
        state.voiceMode = true; // Agent will speak the reply
        window.speechSynthesis.cancel();
      };

      recognition.onresult = (e) => {
        const transcript = e.results[0][0].transcript;
        els.chatInput.value = transcript;
        updateCharCount();
        updateSendBtn();
        sendMessage();
      };

      recognition.onerror = (e) => {
        els.chatInput.placeholder = "Search your documents — ask anything...";
        els.voiceBtn.classList.remove("listening");
        toast("Voice recognition failed: " + e.error, "warn");
      };

      recognition.onend = () => {
        els.voiceBtn.classList.remove("listening");
        els.chatInput.placeholder = "Search your documents — ask anything...";
      };

      els.voiceBtn.addEventListener("click", () => {
        if (els.voiceBtn.classList.contains("listening")) {
          recognition.stop();
        } else {
          try {
            recognition.start();
          } catch (err) {
            toast("Microphone access denied or error: " + err.message, "error");
          }
        }
      });
    } else {
      els.voiceBtn.style.display = "none";
      console.warn("Speech recognition not supported in this browser.");
    }
  }

  // ── Modal overlay click ───────────────────────────────────────
  els.confirmModal.addEventListener("click", (e) => {
    if (e.target === els.confirmModal) {
      els.confirmModal.hidden = true;
    }
  });

  // ── Topic Search Panel ────────────────────────────────────────
  if (els.topicSearchBtn) {
    els.topicSearchBtn.addEventListener("click", () => {
      openTopicSearch();
    });
  }
  if (els.topicSearchClose) {
    els.topicSearchClose.addEventListener("click", () => {
      closeTopicSearch();
    });
  }
  if (els.topicSearchOverlay) {
    els.topicSearchOverlay.addEventListener("click", (e) => {
      if (e.target === els.topicSearchOverlay) closeTopicSearch();
    });
  }
  if (els.topicSearchGo) {
    els.topicSearchGo.addEventListener("click", () => {
      performTopicSearch();
    });
  }
  if (els.topicSearchInput) {
    els.topicSearchInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        performTopicSearch();
      }
    });
  }
  // Suggestion chips inside topic panel
  els.topicSearchResults?.addEventListener("click", (e) => {
    const chip = e.target.closest(".topic-chip");
    if (chip && chip.dataset.keyword) {
      els.topicSearchInput.value = chip.dataset.keyword;
      performTopicSearch();
    }
  });
}

// ═══════════════════════════════════════════════════════════════════
// TOPIC SEARCH
// ═══════════════════════════════════════════════════════════════════

function openTopicSearch() {
  els.topicSearchOverlay.hidden = false;
  setTimeout(() => els.topicSearchInput.focus(), 100);
}

function closeTopicSearch() {
  els.topicSearchOverlay.hidden = true;
}

async function performTopicSearch() {
  const keyword = els.topicSearchInput.value.trim();
  if (!keyword) {
    toast("Please enter a keyword to search", "info");
    return;
  }

  // Show loading
  els.topicEmptyState.hidden = true;
  els.topicLoading.hidden = false;
  els.topicSearchStats.hidden = true;

  // Clear previous results
  els.topicSearchResults.querySelectorAll(".topic-doc-group, .topic-no-results, .topic-ask-ai-btn").forEach(el => el.remove());

  try {
    const data = await apiGet(`/api/documents/search-topics?keyword=${encodeURIComponent(keyword)}`);
    els.topicLoading.hidden = true;

    if (data.total_results === 0) {
      renderTopicNoResults(keyword);
      return;
    }

    // Show stats
    renderTopicStats(data);

    // Render grouped results
    renderTopicResults(data, keyword);
  } catch (err) {
    els.topicLoading.hidden = true;
    toast(`Topic search failed: ${err.message}`, "error");
  }
}

function renderTopicStats(data) {
  els.topicSearchStats.hidden = false;
  const docCount = data.grouped_by_document?.length || 0;
  els.topicSearchStats.innerHTML = `
    <span class="topic-stat-badge keyword">🔑 "${escapeHtml(data.keyword)}"</span>
    <span class="topic-stat-badge count">📄 ${data.total_results} section${data.total_results !== 1 ? 's' : ''} found</span>
    <span class="topic-stat-badge docs">📁 ${docCount} document${docCount !== 1 ? 's' : ''}</span>
  `;
}

function highlightKeyword(text, keyword) {
  if (!keyword) return escapeHtml(text);
  const escaped = escapeHtml(text);
  const keywordEscaped = escapeHtml(keyword);
  const regex = new RegExp(`(${keywordEscaped.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
  return escaped.replace(regex, '<mark>$1</mark>');
}

function getRelevanceClass(score) {
  if (score >= 0.6) return 'high';
  if (score >= 0.35) return 'mid';
  return 'low';
}

function renderTopicResults(data, keyword) {
  const container = els.topicSearchResults;

  data.grouped_by_document.forEach((docGroup, gIdx) => {
    const group = document.createElement('div');
    group.className = 'topic-doc-group';

    const { icon } = getFileIcon(docGroup.filename);
    const pagesText = docGroup.pages.length > 1
      ? `Pages ${docGroup.pages.join(', ')}`
      : `Page ${docGroup.pages[0]}`;

    group.innerHTML = `
      <div class="topic-doc-header">
        <span class="topic-doc-icon">${icon}</span>
        <span class="topic-doc-name">${escapeHtml(docGroup.filename)}</span>
        <span class="topic-doc-pages">${pagesText}</span>
      </div>
    `;

    docGroup.sections.forEach((section, sIdx) => {
      const card = document.createElement('div');
      card.className = 'topic-result-card';
      card.style.animationDelay = `${(gIdx * 3 + sIdx) * 0.05}s`;

      const relevanceClass = getRelevanceClass(section.relevance_score);
      const relevancePct = (section.relevance_score * 100).toFixed(0);

      card.innerHTML = `
        <div class="topic-card-header">
          <span class="topic-card-num">${section.index}</span>
          <span class="topic-card-page">Page ${section.page}</span>
          ${section.has_exact_match ? '<span class="topic-card-exact">✓ Exact match</span>' : ''}
          <span class="topic-card-relevance ${relevanceClass}">${relevancePct}% relevant</span>
        </div>
        <div class="topic-card-snippet">${highlightKeyword(section.snippet, keyword)}</div>
      `;

      // Click card to ask AI about this section
      card.addEventListener('click', () => {
        const query = `Tell me about "${keyword}" as mentioned on page ${section.page} of ${docGroup.filename}. Here is the relevant section: ${section.snippet}`;
        closeTopicSearch();
        els.chatInput.value = query;
        updateCharCount();
        updateSendBtn();
        sendMessage();
      });

      group.appendChild(card);
    });

    container.appendChild(group);
  });

  // Add "Ask AI" summary button
  const askBtn = document.createElement('button');
  askBtn.className = 'topic-ask-ai-btn';
  askBtn.innerHTML = `
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
      <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
    </svg>
    Ask AI to summarize all "${escapeHtml(keyword)}" sections
  `;
  askBtn.addEventListener('click', () => {
    closeTopicSearch();
    els.chatInput.value = `Summarize everything related to "${keyword}" across all uploaded documents. Include page references.`;
    updateCharCount();
    updateSendBtn();
    sendMessage();
  });
  container.appendChild(askBtn);
}

function renderTopicNoResults(keyword) {
  const container = els.topicSearchResults;
  const noRes = document.createElement('div');
  noRes.className = 'topic-no-results';
  noRes.innerHTML = `
    <div class="topic-no-results-icon">🔍</div>
    <p class="topic-no-results-title">No results for "${escapeHtml(keyword)}"</p>
    <p class="topic-no-results-desc">Try a different keyword, or upload more documents to expand the knowledge base.</p>
  `;
  container.appendChild(noRes);
}

// ═══════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════

async function init() {
  initApp();
}

async function initApp() {
  initEvents();
  await checkHealth();
  await loadDocuments();
  await createSession();
  await loadChatHistory();

  // Periodic health check every 30s
  setInterval(checkHealth, 30000);

  console.log("RAG AI Agent — Frontend initialised ✓");
}

let isLoginMode = true;
function showAuthModal() {
  const modal = document.getElementById("authModal");
  if (modal) modal.hidden = false;
}

document.addEventListener("DOMContentLoaded", () => {
  const authModal = document.getElementById("authModal");
  const authToggleBtn = document.getElementById("authToggleBtn");
  const authSubmitBtn = document.getElementById("authSubmitBtn");
  const authTitle = document.getElementById("authTitle");
  const authSub = document.getElementById("authSub");
  const logoutBtn = document.getElementById("logoutBtn");
  
  if (logoutBtn) {
    logoutBtn.addEventListener("click", () => {
      localStorage.removeItem("token");
      location.reload();
    });
  }

  if (authToggleBtn) {
    authToggleBtn.addEventListener("click", () => {
      isLoginMode = !isLoginMode;
      authTitle.textContent = isLoginMode ? "Login" : "Register";
      authSub.textContent = isLoginMode ? "Sign in to access your agent" : "Create a new account";
      authSubmitBtn.textContent = isLoginMode ? "Login" : "Register";
      authToggleBtn.textContent = isLoginMode ? "Need an account? Register" : "Already have an account? Login";
    });
  }

  if (authSubmitBtn) {
    authSubmitBtn.addEventListener("click", async () => {
      const u = document.getElementById("authUsername").value.trim();
      const p = document.getElementById("authPassword").value.trim();
      if (!u || !p) { toast("Enter username and password", "error"); return; }
      
      try {
        if (isLoginMode) {
          const formData = new URLSearchParams();
          formData.append("username", u);
          formData.append("password", p);
          
          const res = await fetch(`${API}/api/auth/token`, {
            method: "POST",
            headers: {"Content-Type": "application/x-www-form-urlencoded"},
            body: formData
          });
          if (!res.ok) throw new Error("Invalid credentials");
          const data = await res.json();
          localStorage.setItem("token", data.access_token);
          authModal.hidden = true;
          toast("Logged in successfully", "success");
          initApp();
        } else {
          const res = await fetch(`${API}/api/auth/register`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({username: u, password: p})
          });
          if (!res.ok) {
              const err = await res.json();
              throw new Error(err.detail || "Registration failed");
          }
          toast("Registered successfully! Please log in.", "success");
          authToggleBtn.click(); // switch to login
        }
      } catch (e) {
        toast(e.message, "error");
      }
    });
  }
  
  // Check auth on load
  if (!localStorage.getItem("token")) {
    showAuthModal();
  } else {
    init();
  }
});
