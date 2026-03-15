const http  = require("http");
const https = require("https");
const fs    = require("fs");
const path  = require("path");

const envPath = path.join(__dirname, ".env");
if (fs.existsSync(envPath)) {
  fs.readFileSync(envPath, "utf8").split("\n").forEach(line => {
    line = line.trim();
    if (!line || line.startsWith("#")) return;
    const idx = line.indexOf("=");
    if (idx < 0) return;
    const key = line.slice(0, idx).trim();
    const val = line.slice(idx + 1).trim();
    if (key && !process.env[key]) process.env[key] = val;
  });
  console.log("✅  .env loaded");
} else {
  console.log("⚠️  No .env — using system env (Railway)");
}

const PORT     = process.env.PORT || 3000;
const PROVIDER = (process.env.AI_PROVIDER || "groq").toLowerCase();
const STATIC   = __dirname;

const GROQ_TEXT_MODEL   = process.env.GROQ_MODEL        || "llama-3.3-70b-versatile";
const GROQ_VISION_MODEL = process.env.GROQ_VISION_MODEL || "meta-llama/llama-4-scout-17b-16e-instruct";
const GEMINI_MODEL      = process.env.GEMINI_MODEL      || "gemini-2.0-flash";
const OPENROUTER_MODEL  = process.env.OPENROUTER_MODEL  || "google/gemini-2.0-flash-exp:free";

// ─── Modelo de visión para OpenRouter (cuando llega imagen) ───────────────
// Si el modelo configurado es "auto" o uno sin visión, usa gemini-flash que sí tiene visión
const OPENROUTER_VISION_MODEL = process.env.OPENROUTER_VISION_MODEL || "google/gemini-2.0-flash-exp:free";

function messageHasImage(messages) {
  if (!messages || messages.length === 0) return false;
  const last = messages[messages.length - 1];
  return Array.isArray(last.content) && last.content.some(b => b.type === "image");
}

console.log("🔧 Variables de entorno detectadas:");
console.log("   PORT                    :", process.env.PORT               || "❌ no definida");
console.log("   AI_PROVIDER             :", process.env.AI_PROVIDER        || "❌ no definida");
console.log("   OPENROUTER_API_KEY      :", process.env.OPENROUTER_API_KEY ? "✅ existe" : "❌ no definida");
console.log("   OPENROUTER_MODEL        :", OPENROUTER_MODEL);
console.log("   OPENROUTER_VISION_MODEL :", OPENROUTER_VISION_MODEL);
console.log("   GEMINI_API_KEY          :", process.env.GEMINI_API_KEY     ? "✅ existe" : "❌ no definida");
console.log("   GEMINI_MODEL            :", GEMINI_MODEL);
console.log("   OPENAI_API_KEY          :", process.env.OPENAI_API_KEY     ? "✅ existe" : "❌ no definida");
console.log("   OPENAI_MODEL            :", process.env.OPENAI_MODEL       || "gpt-4o (default)");
console.log("   GROQ_API_KEY            :", process.env.GROQ_API_KEY       ? "✅ existe" : "❌ no definida");
console.log("   GROQ_MODEL              :", GROQ_TEXT_MODEL);
console.log("   GROQ_VISION_MODEL       :", GROQ_VISION_MODEL);
console.log("   ANTHROPIC_API_KEY       :", process.env.ANTHROPIC_API_KEY  ? "✅ existe" : "❌ no definida");
console.log("   TAVILY_API_KEY          :", process.env.TAVILY_API_KEY     ? `✅ existe (${process.env.TAVILY_API_KEY.slice(0,8)}...)` : "❌ no definida");

const MIME = {
  ".html":"text/html; charset=utf-8",
  ".css":"text/css; charset=utf-8",
  ".js":"application/javascript; charset=utf-8",
  ".json":"application/json",
  ".png":"image/png",".jpg":"image/jpeg",".jpeg":"image/jpeg",
  ".ico":"image/x-icon",".svg":"image/svg+xml",".webp":"image/webp",
};

function readBody(req) {
  return new Promise((res, rej) => {
    const chunks = [];
    req.on("data", c => chunks.push(c));
    req.on("end", () => res(Buffer.concat(chunks).toString("utf8")));
    req.on("error", rej);
  });
}

// ─── Convierte content al formato OpenAI/OpenRouter ──────────────────────
function toOpenAIContent(content, stripImages = false) {
  if (!content) return null;
  if (typeof content === "string") return content.trim() || null;
  if (Array.isArray(content)) {
    const parts = [];
    for (const block of content) {
      if (!block || !block.type) continue;
      if (block.type === "text") {
        const t = (block.text || "").trim();
        if (t) parts.push({ type: "text", text: t });
      } else if (block.type === "image") {
        if (stripImages) {
          parts.push({ type: "text", text: "[imagen adjunta]" });
        } else {
          const src = block.source || {};
          if (src.type === "base64" && src.data) {
            const mime = src.media_type || "image/png";
            // ✅ CORREGIDO: formato correcto image_url para OpenRouter/OpenAI
            parts.push({ type: "image_url", image_url: { url: `data:${mime};base64,${src.data}` } });
          } else if (src.type === "url" && src.url) {
            parts.push({ type: "image_url", image_url: { url: src.url } });
          }
        }
      }
    }
    if (parts.length === 0) return null;
    if (parts.length === 1 && parts[0].type === "text") return parts[0].text;
    return parts;
  }
  return String(content).trim() || null;
}

function contentToString(content) {
  if (!content) return null;
  if (typeof content === "string") return content.trim() || null;
  if (Array.isArray(content)) {
    const parts = [];
    for (const b of content) {
      if (!b) continue;
      if (b.type === "text" && b.text?.trim()) parts.push(b.text.trim());
      else if (b.type === "image" || b.type === "image_url") parts.push("[imagen]");
    }
    return parts.join("\n") || null;
  }
  return String(content).trim() || null;
}

function flattenForOpenAI(messages, systemPrompt, useVision = true) {
  const result = [];
  if (systemPrompt) result.push({ role: "system", content: systemPrompt });
  let lastRole = "system";
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    const role = m.role === "assistant" ? "assistant" : "user";
    const isLast = (i === messages.length - 1);
    const stripImages = !useVision;
    let content = isLast && role === "user"
      ? toOpenAIContent(m.content, stripImages)
      : contentToString(m.content);
    if (!content) continue;
    if (role === lastRole) {
      const prev = result[result.length - 1];
      if (typeof prev.content === "string" && typeof content === "string") {
        prev.content += "\n" + content;
      } else {
        const toArr = c => typeof c === "string"
          ? [{ type: "text", text: c }]
          : (Array.isArray(c) ? c : [{ type: "text", text: String(c) }]);
        prev.content = [...toArr(prev.content), ...toArr(content)];
      }
    } else {
      result.push({ role, content });
      lastRole = role;
    }
  }
  const hasUser = result.some(m => m.role === "user");
  if (!hasUser) return null;
  return result;
}

// ─── Convierte mensajes al formato Gemini ────────────────────────────────
function toGeminiMessages(messages, systemPrompt) {
  const contents = [];
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    const role = m.role === "assistant" ? "model" : "user";
    const isLast = (i === messages.length - 1);
    const parts = [];
    if (i === 0 && systemPrompt && role === "user") {
      parts.push({ text: systemPrompt + "\n\n" });
    }
    if (typeof m.content === "string") {
      if (m.content.trim()) parts.push({ text: m.content.trim() });
    } else if (Array.isArray(m.content)) {
      for (const block of m.content) {
        if (!block || !block.type) continue;
        if (block.type === "text" && block.text?.trim()) {
          parts.push({ text: block.text.trim() });
        } else if (block.type === "image" && isLast) {
          const src = block.source || {};
          if (src.type === "base64" && src.data) {
            parts.push({ inlineData: { mimeType: src.media_type || "image/png", data: src.data } });
          }
        }
      }
    }
    if (parts.length === 0) continue;
    if (contents.length > 0 && contents[contents.length - 1].role === role) {
      contents[contents.length - 1].parts.push(...parts);
    } else {
      contents.push({ role, parts });
    }
  }
  while (contents.length > 0 && contents[0].role !== "user") contents.shift();
  if (contents.length === 0) return null;
  return contents;
}

function flattenForAnthropic(messages) {
  const result = [];
  let lastRole = null;
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    const role = m.role === "assistant" ? "assistant" : "user";
    const isLast = i === messages.length - 1;
    let content = m.content;
    if (!content) continue;
    if (isLast && role === "user") {
      if (Array.isArray(content)) {
        const blocks = content.filter(b => {
          if (!b || !b.type) return false;
          if (b.type === "text") return b.text?.trim();
          if (b.type === "image") return b.source?.data;
          return false;
        });
        if (blocks.length === 0) continue;
        content = blocks.every(b => b.type === "text")
          ? blocks.map(b => b.text.trim()).join("\n")
          : blocks;
      } else if (typeof content === "string") {
        content = content.trim();
        if (!content) continue;
      }
    } else {
      if (typeof content === "string") {
        content = content.trim();
        if (!content) continue;
      } else if (Array.isArray(content)) {
        const text = content
          .filter(b => b && b.type === "text" && b.text?.trim())
          .map(b => b.text.trim()).join("\n").trim();
        if (!text) continue;
        content = text;
      } else {
        content = String(content).trim();
        if (!content) continue;
      }
    }
    if (role === lastRole) {
      const prev = result[result.length - 1];
      const toStr = c => typeof c === "string"
        ? c : c.filter(b => b.type === "text").map(b => b.text).join("\n");
      prev.content = toStr(prev.content) + "\n" + toStr(content);
    } else {
      result.push({ role, content });
      lastRole = role;
    }
  }
  while (result.length > 0 && result[0].role !== "user") result.shift();
  if (result.length === 0) return null;
  return result;
}

// ─── Detecta si necesita búsqueda web ────────────────────────────────────
function needsWebSearch(messages) {
  if (!messages || messages.length === 0) return false;
  const last = messages[messages.length - 1];
  const text = typeof last.content === "string"
    ? last.content
    : (Array.isArray(last.content)
        ? last.content.filter(b => b.type === "text").map(b => b.text).join(" ")
        : "");
  const keywords = [
    "actualidad","actual","ahora","hoy","2024","2025","2026","2027",
    "noticia","noticias","último","ultima","últimas","ultimas",
    "reciente","recientes","recientemente","hoy en dia","actualmente",
    "precio","cotización","dolar","bitcoin","crypto","bolsa","mercado",
    "gol","goles","partido","resultado","campeón","campeon","liga",
    "mundial","torneo","clasificacion","tabla","standings",
    "presidente","elección","elecciones","guerra","crisis","gobierno",
    "ministro","congreso","senado","parlamento",
    "lanzó","lanzamiento","estreno","nuevo modelo","nueva version",
    "película","pelicula","serie","temporada","anime","manga",
    "todos los","todas las","lista de","lista completa","cuántos hay",
    "cuantos hay","historia de","evolución","evolution",
    "sentai","kamen rider","ultraman","marvel","dc comics",
    "cuánto va","cuanto va","quién ganó","quien gano","quién es",
    "quien es el actual","temperatura","clima","tiempo en",
    "cuántos","cuantos","cuántas","cuantas",
    "dame todos","dame todas","enumera","lista"
  ];
  return keywords.some(k => text.toLowerCase().includes(k));
}

function extractLastUserText(messages) {
  if (!messages || messages.length === 0) return "";
  const last = messages[messages.length - 1];
  if (typeof last.content === "string") return last.content;
  if (Array.isArray(last.content))
    return last.content.filter(b => b.type === "text").map(b => b.text).join(" ");
  return "";
}

// ─── Búsqueda con Tavily ──────────────────────────────────────────────────
async function doTavilySearch(query) {
  const apiKey = process.env.TAVILY_API_KEY;
  if (!apiKey) { console.log("❌ TAVILY_API_KEY no definida"); return null; }
  console.log(`🔍 Tavily: "${query.slice(0, 80)}..."`);
  const body = JSON.stringify({
    api_key: apiKey, query: query.slice(0, 400),
    search_depth: "advanced", include_answer: true,
    include_raw_content: false, max_results: 5,
    include_domains: [], exclude_domains: []
  });
  return new Promise((resolve) => {
    const req = https.request({
      hostname: "api.tavily.com", path: "/search", method: "POST",
      headers: { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(body) }
    }, res => {
      const chunks = [];
      res.on("data", c => chunks.push(c));
      res.on("end", () => {
        try {
          const data = JSON.parse(Buffer.concat(chunks).toString("utf8"));
          let result = "";
          if (data.answer) result += `Respuesta directa: ${data.answer}\n\n`;
          if (data.results?.length)
            result += data.results.map((r,i) => `[${i+1}] ${r.title}\n${r.content||r.snippet||""}`).join("\n\n");
          if (!result.trim()) return resolve(null);
          console.log(`✅ Tavily: ${result.length} chars`);
          resolve(result);
        } catch(e) { console.log(`❌ Tavily error: ${e.message}`); resolve(null); }
      });
    });
    req.on("error", e => { console.log(`❌ Tavily error: ${e.message}`); resolve(null); });
    req.write(body); req.end();
  });
}

function injectSearchResults(systemPrompt, searchResults) {
  const today = new Date().toLocaleDateString("es-ES", {
    weekday: "long", year: "numeric", month: "long", day: "numeric"
  });
  return systemPrompt + `

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🌐 BÚSQUEDA WEB EN TIEMPO REAL
Fecha actual: ${today}

Los siguientes resultados fueron obtenidos de internet ahora mismo. Úsalos para dar una respuesta actualizada y precisa. NO incluyas URLs ni links en tu respuesta a menos que el usuario los pida explícitamente.

${searchResults}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`;
}

async function callAPI(payload) {
  const rawMessages = payload.messages || [];
  let systemPrompt = payload.system || "";
  let hostname, urlPath, headers, body;

  if (needsWebSearch(rawMessages)) {
    const query = extractLastUserText(rawMessages);
    const searchResults = await doTavilySearch(query);
    if (searchResults) {
      systemPrompt = injectSearchResults(systemPrompt, searchResults);
      console.log("✅ Tavily inyectado");
    } else {
      const today = new Date().toLocaleDateString("es-ES", {
        weekday: "long", year: "numeric", month: "long", day: "numeric"
      });
      systemPrompt += `\n\n⚠️ Fecha actual: ${today}. NUNCA inventes cifras recientes.`;
    }
  }

  // ── OPENROUTER ────────────────────────────────────────────────────────────
  if (PROVIDER === "openrouter") {
    const apiKey = process.env.OPENROUTER_API_KEY;
    if (!apiKey) throw new Error("OPENROUTER_API_KEY no definida");

    const hasImage = messageHasImage(rawMessages);

    // ✅ CORRECCIÓN PRINCIPAL:
    // Si hay imagen, forzamos un modelo con visión garantizada.
    // "openrouter/auto" puede elegir un modelo sin visión y devolver respuesta vacía.
    const model = hasImage ? OPENROUTER_VISION_MODEL : OPENROUTER_MODEL;

    const msgs = flattenForOpenAI(rawMessages, systemPrompt, true);
    if (!msgs) throw new Error("No hay mensajes válidos.");

    body = JSON.stringify({
      model,
      messages: msgs,
      max_tokens: 4000,
      temperature: 0.3,
    });

    hostname = "openrouter.ai";
    urlPath  = "/api/v1/chat/completions";
    headers  = {
      "Content-Type":   "application/json",
      "Authorization":  `Bearer ${apiKey}`,
      "HTTP-Referer":   "https://kenyra.ia",
      "X-Title":        "Kenyra IA",
      "Content-Length": Buffer.byteLength(body),
    };
    console.log(`\n→ OPENROUTER [${model}] imagen:${hasImage} msgs:${msgs.length} body:${body.length}b`);

  // ── GEMINI ────────────────────────────────────────────────────────────────
  } else if (PROVIDER === "gemini") {
    const model  = GEMINI_MODEL;
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) throw new Error("GEMINI_API_KEY no definida");
    const contents = toGeminiMessages(rawMessages, systemPrompt);
    if (!contents) throw new Error("No hay mensajes válidos.");
    body     = JSON.stringify({ contents, generationConfig: { maxOutputTokens: 4000, temperature: 0.3 } });
    hostname = "generativelanguage.googleapis.com";
    urlPath  = `/v1beta/models/${model}:generateContent?key=${apiKey}`;
    headers  = { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(body) };
    console.log(`\n→ GEMINI [${model}] msgs:${contents.length} body:${body.length}b`);

  // ── OPENAI ────────────────────────────────────────────────────────────────
  } else if (PROVIDER === "openai") {
    const model = process.env.OPENAI_MODEL || "gpt-4o";
    const msgs  = flattenForOpenAI(rawMessages, systemPrompt, true);
    if (!msgs) throw new Error("No hay mensajes válidos.");
    body     = JSON.stringify({ model, messages: msgs, max_tokens: 4000 });
    hostname = "api.openai.com";
    urlPath  = "/v1/chat/completions";
    headers  = {
      "Content-Type":   "application/json",
      "Authorization":  `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Length": Buffer.byteLength(body),
    };
    console.log(`\n→ OPENAI [${model}] msgs:${msgs.length} body:${body.length}b`);

  // ── GROQ ──────────────────────────────────────────────────────────────────
  } else if (PROVIDER === "groq") {
    const hasImage = messageHasImage(rawMessages);
    const model    = hasImage ? GROQ_VISION_MODEL : GROQ_TEXT_MODEL;
    const msgs     = flattenForOpenAI(rawMessages, systemPrompt, hasImage);
    if (!msgs) throw new Error("No hay mensajes válidos.");
    body     = JSON.stringify({ model, messages: msgs, max_tokens: 4000, temperature: 0.3 });
    hostname = "api.groq.com";
    urlPath  = "/openai/v1/chat/completions";
    headers  = {
      "Content-Type":   "application/json",
      "Authorization":  `Bearer ${process.env.GROQ_API_KEY}`,
      "Content-Length": Buffer.byteLength(body),
    };
    console.log(`\n→ GROQ [${model}] imagen:${hasImage} msgs:${msgs.length} body:${body.length}b`);

  // ── ANTHROPIC ─────────────────────────────────────────────────────────────
  } else if (PROVIDER === "anthropic") {
    const model = process.env.ANTHROPIC_MODEL || "claude-sonnet-4-20250514";
    const msgs  = flattenForAnthropic(rawMessages);
    if (!msgs) throw new Error("No hay mensajes válidos.");
    const apiPayload = {
      model, max_tokens: 4000, messages: msgs,
      tools: [{ type: "web_search_20250305", name: "web_search" }]
    };
    if (systemPrompt) apiPayload.system = systemPrompt;
    body     = JSON.stringify(apiPayload);
    hostname = "api.anthropic.com";
    urlPath  = "/v1/messages";
    headers  = {
      "Content-Type":      "application/json",
      "x-api-key":         process.env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
      "anthropic-beta":    "web-search-2025-03-05",
      "Content-Length":    Buffer.byteLength(body),
    };
    console.log(`\n→ ANTHROPIC [${model}] msgs:${msgs.length} body:${body.length}b`);

  } else {
    throw new Error(`Proveedor desconocido: ${PROVIDER}`);
  }

  return new Promise((resolve, reject) => {
    const req = https.request({ hostname, path: urlPath, method: "POST", headers }, res => {
      const chunks = [];
      res.on("data", c => chunks.push(c));
      res.on("end", () => {
        const raw = Buffer.concat(chunks).toString("utf8");
        console.log(`← ${PROVIDER.toUpperCase()} [${res.statusCode}]: ${raw.slice(0, 400)}`);
        resolve({ status: res.statusCode, body: raw });
      });
    });
    req.on("error", reject);
    req.write(body);
    req.end();
  });
}

function normalizeResponse(raw, status) {
  // ── Gemini ──
  if (PROVIDER === "gemini") {
    try {
      const data = JSON.parse(raw);
      if (status !== 200) {
        return JSON.stringify({ content: [{ type: "text", text: "Error API: " + (data.error?.message || raw.slice(0,200)) }] });
      }
      const text = data.candidates?.[0]?.content?.parts?.filter(p => p.text)?.map(p => p.text)?.join("") || "";
      return JSON.stringify({ content: [{ type: "text", text }] });
    } catch {
      return JSON.stringify({ content: [{ type: "text", text: "Error al parsear respuesta Gemini." }] });
    }
  }
  // ── Anthropic ──
  if (PROVIDER === "anthropic") {
    if (status !== 200) {
      try {
        const e = JSON.parse(raw);
        return JSON.stringify({ content: [{ type: "text", text: "Error API: " + (e.error?.message || raw.slice(0,200)) }] });
      } catch { return raw; }
    }
    try {
      const data = JSON.parse(raw);
      const textBlocks = (data.content || []).filter(b => b.type === "text");
      if (textBlocks.length > 0) return JSON.stringify({ content: textBlocks });
      return raw;
    } catch { return raw; }
  }
  // ── OpenRouter / OpenAI / Groq ──
  try {
    const data = JSON.parse(raw);
    if (status !== 200) {
      const msg = data.error?.message || raw.slice(0, 200);
      return JSON.stringify({ content: [{ type: "text", text: "Error API: " + msg }] });
    }
    const text = data.choices?.[0]?.message?.content || "";
    return JSON.stringify({ content: [{ type: "text", text }] });
  } catch {
    return JSON.stringify({ content: [{ type: "text", text: "Error al parsear respuesta." }] });
  }
}

const server = http.createServer(async (req, res) => {
  const parsed = new URL(req.url, `http://localhost:${PORT}`);

  res.setHeader("Access-Control-Allow-Origin",  "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.writeHead(204).end();

  if (req.method === "POST" && parsed.pathname === "/api/chat") {
    try {
      const rawBody = await readBody(req);
      const payload = JSON.parse(rawBody);
      const { status, body } = await callAPI(payload);
      const normalized = normalizeResponse(body, status);
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(normalized);
    } catch (err) {
      console.error("❌ Internal error:", err.message);
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ content: [{ type: "text", text: "Error interno: " + err.message }] }));
    }
    return;
  }

  let filePath = path.join(STATIC, parsed.pathname === "/" ? "index.html" : parsed.pathname);
  const ext = path.extname(filePath);
  if (!ext) filePath += ".html";

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { "Content-Type": "text/plain" });
      return res.end("404 — File not found");
    }
    res.writeHead(200, { "Content-Type": MIME[ext] || "application/octet-stream" });
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log("\n╔══════════════════════════════════════════════════╗");
  console.log("║           🤖  KENYRA IA  🤖                      ║");
  console.log("╠══════════════════════════════════════════════════╣");
  console.log(`║  Puerto:    ${PORT}                                   ║`);
  console.log(`║  Provider:  ${PROVIDER.toUpperCase().padEnd(37)}║`);
  if (PROVIDER === "openrouter") {
    console.log(`║  Texto:     ${OPENROUTER_MODEL.slice(0,37).padEnd(37)}║`);
    console.log(`║  Visión:    ${OPENROUTER_VISION_MODEL.slice(0,37).padEnd(37)}║`);
  } else if (PROVIDER === "gemini") {
    console.log(`║  Modelo:    ${GEMINI_MODEL.padEnd(37)}║`);
    console.log(`║  Vision:    ✅ nativa (Gemini)                   ║`);
  } else if (PROVIDER === "openai") {
    console.log(`║  Modelo:    ${(process.env.OPENAI_MODEL||"gpt-4o").padEnd(37)}║`);
    console.log(`║  Vision:    ✅ nativa (GPT-4o)                   ║`);
  } else if (PROVIDER === "groq") {
    console.log(`║  Texto:     ${GROQ_TEXT_MODEL.padEnd(37)}║`);
    console.log(`║  Vision:    ${GROQ_VISION_MODEL.slice(0,37).padEnd(37)}║`);
  } else if (PROVIDER === "anthropic") {
    console.log(`║  Modelo:    ${(process.env.ANTHROPIC_MODEL||"claude-sonnet").padEnd(37)}║`);
    console.log(`║  Vision:    ✅ nativa (Anthropic)                ║`);
  }
  console.log(`║  Tavily:    ${process.env.TAVILY_API_KEY ? "✅ activo" : "⚠️  no configurado"}                        ║`);
  console.log("╚══════════════════════════════════════════════════╝\n");
});

server.on("error", err => {
  if (err.code === "EADDRINUSE") console.error(`❌ Puerto ${PORT} en uso.`);
  else console.error("❌ Error:", err.message);
  process.exit(1);
});
