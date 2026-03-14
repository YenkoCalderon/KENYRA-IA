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

function toOpenAIContent(content) {
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
        const src = block.source || {};
        if (src.type === "base64" && src.data) {
          const mime = src.media_type || "image/png";
          parts.push({ type: "image_url", image_url: { url: `data:${mime};base64,${src.data}` } });
        } else if (src.type === "url" && src.url) {
          parts.push({ type: "image_url", image_url: { url: src.url } });
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

function flattenForOpenAI(messages, systemPrompt) {
  const result = [];
  if (systemPrompt) result.push({ role: "system", content: systemPrompt });
  let lastRole = "system";
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    const role = m.role === "assistant" ? "assistant" : "user";
    const isLast = (i === messages.length - 1);
    let content = isLast && role === "user" ? toOpenAIContent(m.content) : contentToString(m.content);
    if (!content) continue;
    if (role === lastRole) {
      const prev = result[result.length - 1];
      if (typeof prev.content === "string" && typeof content === "string") {
        prev.content += "\n" + content;
      } else {
        const toArr = c => typeof c === "string" ? [{ type: "text", text: c }] : (Array.isArray(c) ? c : [{ type: "text", text: String(c) }]);
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

function flattenForAnthropic(messages) {
  const result = [];
  let lastRole = null;
  for (const m of messages) {
    const role = m.role === "assistant" ? "assistant" : "user";
    let content = m.content;
    if (!content) continue;
    if (typeof content === "string") {
      content = content.trim();
      if (!content) continue;
    } else if (Array.isArray(content)) {
      content = content.filter(b => {
        if (!b || !b.type) return false;
        if (b.type === "text") return typeof b.text === "string" && b.text.trim();
        if (b.type === "image") return b.source && b.source.data;
        return false;
      });
      if (content.length === 0) continue;
      if (content.every(b => b.type === "text")) content = content.map(b => b.text).join("\n");
    } else {
      content = String(content).trim();
      if (!content) continue;
    }
    if (role === lastRole) {
      const prev = result[result.length - 1];
      const toStr = c => typeof c === "string" ? c : c.filter(b => b.type === "text").map(b => b.text).join("\n");
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

// ─── NUEVA FUNCIÓN: Detecta si el mensaje necesita búsqueda web ───────────
function needsWebSearch(messages) {
  if (!messages || messages.length === 0) return false;
  const last = messages[messages.length - 1];
  const text = typeof last.content === "string"
    ? last.content
    : (Array.isArray(last.content) ? last.content.filter(b => b.type === "text").map(b => b.text).join(" ") : "");
  const keywords = [
    "actualidad", "actual", "ahora", "hoy", "2025", "2026", "2027",
    "noticia", "noticias", "último", "ultima", "últimas", "ultimas",
    "reciente", "recientes", "precio", "cotización", "dolar", "bitcoin",
    "gol", "goles", "partido", "resultado", "campeón", "campeon",
    "presidente", "elección", "elecciones", "guerra", "crisis",
    "lanzó", "lanzamiento", "estreno", "nuevo modelo", "nueva version",
    "cuánto va", "cuanto va", "quién ganó", "quien gano", "quién es",
    "quien es el actual", "temperatura", "clima", "tiempo en"
  ];
  const lower = text.toLowerCase();
  return keywords.some(k => lower.includes(k));
}

// ─── NUEVA FUNCIÓN: Hace búsqueda web real con Brave Search API ──────────
async function doWebSearch(query) {
  const apiKey = process.env.BRAVE_API_KEY;
  if (!apiKey) return null;

  return new Promise((resolve) => {
    const q = encodeURIComponent(query.slice(0, 200));
    const options = {
      hostname: "api.search.brave.com",
      path: `/res/v1/web/search?q=${q}&count=5&country=PE&search_lang=es`,
      method: "GET",
      headers: {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": apiKey
      }
    };
    const req = https.request(options, res => {
      const chunks = [];
      res.on("data", c => chunks.push(c));
      res.on("end", () => {
        try {
          const data = JSON.parse(Buffer.concat(chunks).toString("utf8"));
          const results = (data.web?.results || []).slice(0, 5);
          if (!results.length) return resolve(null);
          const summary = results.map((r, i) =>
            `[${i+1}] ${r.title}\n${r.description || ""}\nURL: ${r.url}`
          ).join("\n\n");
          resolve(summary);
        } catch { resolve(null); }
      });
    });
    req.on("error", () => resolve(null));
    req.end();
  });
}

// ─── NUEVA FUNCIÓN: Extrae el texto principal del último mensaje ──────────
function extractLastUserText(messages) {
  if (!messages || messages.length === 0) return "";
  const last = messages[messages.length - 1];
  if (typeof last.content === "string") return last.content;
  if (Array.isArray(last.content)) {
    return last.content.filter(b => b.type === "text").map(b => b.text).join(" ");
  }
  return "";
}

// ─── NUEVA FUNCIÓN: Inyecta resultados de búsqueda en el system prompt ───
function injectSearchResults(systemPrompt, searchResults) {
  const today = new Date().toLocaleDateString("es-ES", {
    weekday: "long", year: "numeric", month: "long", day: "numeric"
  });
  return systemPrompt + `

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🌐 BÚSQUEDA WEB EN TIEMPO REAL
Fecha actual: ${today}

Los siguientes resultados fueron obtenidos de internet ahora mismo para responder la pregunta del usuario. Úsalos para dar una respuesta actualizada y precisa. Cita las fuentes cuando sea relevante.

${searchResults}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`;
}

async function callAPI(payload) {
  return new Promise(async (resolve, reject) => {
    const rawMessages = payload.messages || [];
    let systemPrompt = payload.system || "";
    let hostname, urlPath, headers, body;

    // ── Búsqueda web automática (si hay BRAVE_API_KEY configurada) ─────────
    if (needsWebSearch(rawMessages)) {
      const query = extractLastUserText(rawMessages);
      console.log(`🔍 Buscando en web: "${query.slice(0, 80)}..."`);
      const searchResults = await doWebSearch(query);
      if (searchResults) {
        systemPrompt = injectSearchResults(systemPrompt, searchResults);
        console.log("✅ Resultados web inyectados en el prompt");
      } else {
        // Sin Brave API Key: inyectar fecha actual para que el modelo sea consciente
        const today = new Date().toLocaleDateString("es-ES", {
          weekday: "long", year: "numeric", month: "long", day: "numeric"
        });
        systemPrompt += `\n\n⚠️ Fecha actual: ${today}. Si el usuario pregunta sobre datos en tiempo real (estadísticas, noticias, precios), indica que no tienes acceso a internet en este momento y que los datos pueden estar desactualizados. NUNCA inventes cifras o estadísticas recientes.`;
        console.log("⚠️  Sin BRAVE_API_KEY — fecha inyectada, sin búsqueda web");
      }
    }

    if (PROVIDER === "groq") {
      const model = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";
      const msgs = flattenForOpenAI(rawMessages, systemPrompt);
      if (!msgs) return reject(new Error("No hay mensajes válidos."));
      body = JSON.stringify({ model, messages: msgs, max_tokens: 1500, temperature: 0.7 });
      hostname = "api.groq.com";
      urlPath  = "/openai/v1/chat/completions";
      headers  = {
        "Content-Type":   "application/json",
        "Authorization":  `Bearer ${process.env.GROQ_API_KEY}`,
        "Content-Length": Buffer.byteLength(body),
      };
      console.log(`\n→ GROQ [${model}] msgs:${msgs.length} body:${body.length}b`);

    } else if (PROVIDER === "anthropic") {
      const model = process.env.ANTHROPIC_MODEL || "claude-sonnet-4-20250514";
      const msgs = flattenForAnthropic(rawMessages);
      if (!msgs) return reject(new Error("No hay mensajes válidos."));

      const apiPayload = {
        model,
        max_tokens: 1500,
        messages: msgs,
        // ── Web Search nativo de Anthropic ────────────────────────────────
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
        "Content-Length":    Buffer.byteLength(body),
      };
      console.log(`\n→ ANTHROPIC [${model}] msgs:${msgs.length} body:${body.length}b (web_search activo)`);

    } else {
      const model = process.env.OPENAI_MODEL || "gpt-4o";
      const msgs = flattenForOpenAI(rawMessages, systemPrompt);
      if (!msgs) return reject(new Error("No hay mensajes válidos."));
      body     = JSON.stringify({ model, messages: msgs, max_tokens: 1500 });
      hostname = "api.openai.com";
      urlPath  = "/v1/chat/completions";
      headers  = {
        "Content-Type":   "application/json",
        "Authorization":  `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Length": Buffer.byteLength(body),
      };
      console.log(`\n→ OPENAI [${model}] msgs:${msgs.length} body:${body.length}b`);
    }

    const req = https.request({ hostname, path: urlPath, method: "POST", headers }, res => {
      const chunks = [];
      res.on("data", c => chunks.push(c));
      res.on("end", () => {
        const raw = Buffer.concat(chunks).toString("utf8");
        console.log(`← ${PROVIDER.toUpperCase()} [${res.statusCode}]: ${raw.slice(0, 300)}`);
        resolve({ status: res.statusCode, body: raw });
      });
    });
    req.on("error", reject);
    req.write(body);
    req.end();
  });
}

function normalizeResponse(raw, status) {
  if (PROVIDER === "anthropic") {
    if (status !== 200) {
      try {
        const e = JSON.parse(raw);
        const msg = e.error?.message || raw.slice(0, 200);
        return JSON.stringify({ content: [{ type: "text", text: "Error API: " + msg }] });
      } catch { return raw; }
    }
    // ── Extraer solo bloques de texto (ignorar tool_use y tool_result) ────
    try {
      const data = JSON.parse(raw);
      const textBlocks = (data.content || []).filter(b => b.type === "text");
      if (textBlocks.length > 0) {
        return JSON.stringify({ content: textBlocks });
      }
      return raw;
    } catch { return raw; }
  }
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
  console.log("\n╔══════════════════════════════════════╗");
  console.log("║       🤖  KENYRA IA  🤖              ║");
  console.log("╠══════════════════════════════════════╣");
  console.log(`║  Puerto:    ${PORT}                      ║`);
  console.log(`║  Provider:  ${PROVIDER.toUpperCase().padEnd(26)}║`);
  console.log(`║  Web Search: ${process.env.BRAVE_API_KEY ? "✅ Brave API" : "⚠️  Sin Brave API Key"}        ║`);
  console.log("╚══════════════════════════════════════╝\n");
});

server.on("error", err => {
  if (err.code === "EADDRINUSE") console.error(`❌ Puerto ${PORT} en uso.`);
  else console.error("❌ Error:", err.message);
  process.exit(1);
});
