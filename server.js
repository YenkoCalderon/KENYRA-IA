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

// ─── Modelos Groq ──────────────────────────────────────────────────────────
// GROQ_MODEL        = modelo de texto/razonamiento (ej: llama-3.3-70b-versatile)
// GROQ_VISION_MODEL = modelo con visión (ej: meta-llama/llama-4-scout-17b-16e-instruct)
const GROQ_TEXT_MODEL   = process.env.GROQ_MODEL        || "llama-3.3-70b-versatile";
const GROQ_VISION_MODEL = process.env.GROQ_VISION_MODEL || "meta-llama/llama-4-scout-17b-16e-instruct";

// Detecta si el mensaje actual contiene imágenes
function messageHasImage(messages) {
  if (!messages || messages.length === 0) return false;
  const last = messages[messages.length - 1];
  return Array.isArray(last.content) && last.content.some(b => b.type === "image");
}

console.log("🔧 Variables de entorno detectadas:");
console.log("   PORT             :", process.env.PORT        || "❌ no definida");
console.log("   AI_PROVIDER      :", process.env.AI_PROVIDER || "❌ no definida");
console.log("   GROQ_API_KEY     :", process.env.GROQ_API_KEY  ? "✅ existe" : "❌ no definida");
console.log("   GROQ_MODEL       :", GROQ_TEXT_MODEL);
console.log("   GROQ_VISION_MODEL:", GROQ_VISION_MODEL);
console.log("   TAVILY_API_KEY   :", process.env.TAVILY_API_KEY ? `✅ existe (${process.env.TAVILY_API_KEY.slice(0,8)}...)` : "❌ no definida");

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

// ─── Convierte content al formato OpenAI ─────────────────────────────────
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
          parts.push({ type: "text", text: "[imagen adjunta — modelo sin visión]" });
        } else {
          const src = block.source || {};
          if (src.type === "base64" && src.data) {
            const mime = src.media_type || "image/png";
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

function flattenForOpenAI(messages, systemPrompt, useVision = false) {
  const result = [];
  if (systemPrompt) result.push({ role: "system", content: systemPrompt });
  let lastRole = "system";
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    const role = m.role === "assistant" ? "assistant" : "user";
    const isLast = (i === messages.length - 1);
    // Solo pasar imágenes si estamos usando el modelo vision
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
          .map(b => b.text.trim())
          .join("\n")
          .trim();
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
        ? c
        : c.filter(b => b.type === "text").map(b => b.text).join("\n");
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

// ─── Detecta si el mensaje necesita búsqueda web ─────────────────────────
function needsWebSearch(messages) {
  if (!messages || messages.length === 0) return false;
  const last = messages[messages.length - 1];
  const text = typeof last.content === "string"
    ? last.content
    : (Array.isArray(last.content)
        ? last.content.filter(b => b.type === "text").map(b => b.text).join(" ")
        : "");
  const keywords = [
    "actualidad", "actual", "ahora", "hoy", "2024", "2025", "2026", "2027",
    "noticia", "noticias", "último", "ultima", "últimas", "ultimas",
    "reciente", "recientes", "recientemente", "hoy en dia", "actualmente",
    "precio", "cotización", "dolar", "bitcoin", "crypto", "bolsa", "mercado",
    "gol", "goles", "partido", "resultado", "campeón", "campeon", "liga",
    "mundial", "torneo", "clasificacion", "tabla", "standings",
    "presidente", "elección", "elecciones", "guerra", "crisis", "gobierno",
    "ministro", "congreso", "senado", "parlamento",
    "lanzó", "lanzamiento", "estreno", "nuevo modelo", "nueva version",
    "película", "pelicula", "serie", "temporada", "anime", "manga",
    "todos los", "todas las", "lista de", "lista completa", "cuántos hay",
    "cuantos hay", "historia de", "evolución", "evolution",
    "sentai", "kamen rider", "ultraman", "marvel", "dc comics",
    "cuánto va", "cuanto va", "quién ganó", "quien gano", "quién es",
    "quien es el actual", "temperatura", "clima", "tiempo en",
    "cuántos", "cuantos", "cuántas", "cuantas",
    "dame todos", "dame todas", "enumera", "lista"
  ];
  const lower = text.toLowerCase();
  return keywords.some(k => lower.includes(k));
}

function extractLastUserText(messages) {
  if (!messages || messages.length === 0) return "";
  const last = messages[messages.length - 1];
  if (typeof last.content === "string") return last.content;
  if (Array.isArray(last.content)) {
    return last.content.filter(b => b.type === "text").map(b => b.text).join(" ");
  }
  return "";
}

// ─── Búsqueda con Tavily ──────────────────────────────────────────────────
async function doTavilySearch(query) {
  const apiKey = process.env.TAVILY_API_KEY;
  if (!apiKey) { console.log("❌ TAVILY_API_KEY no definida"); return null; }

  console.log(`🔍 Tavily buscando: "${query.slice(0, 80)}..."`);

  const body = JSON.stringify({
    api_key: apiKey,
    query: query.slice(0, 400),
    search_depth: "advanced",
    include_answer: true,
    include_raw_content: false,
    max_results: 5,
    include_domains: [],
    exclude_domains: []
  });

  return new Promise((resolve) => {
    const req = https.request({
      hostname: "api.tavily.com",
      path: "/search",
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(body)
      }
    }, res => {
      const chunks = [];
      res.on("data", c => chunks.push(c));
      res.on("end", () => {
        try {
          const raw = Buffer.concat(chunks).toString("utf8");
          console.log(`🔎 Tavily [${res.statusCode}]: ${raw.slice(0, 200)}`);
          const data = JSON.parse(raw);
          let result = "";
          if (data.answer) result += `Respuesta directa: ${data.answer}\n\n`;
          if (data.results?.length) {
            result += data.results.map((r, i) =>
              `[${i+1}] ${r.title}\n${r.content || r.snippet || ""}`
            ).join("\n\n");
          }
          if (!result.trim()) return resolve(null);
          console.log(`✅ Tavily: ${result.length} chars`);
          resolve(result);
        } catch(e) {
          console.log(`❌ Tavily parse error: ${e.message}`);
          resolve(null);
        }
      });
    });
    req.on("error", e => { console.log(`❌ Tavily error: ${e.message}`); resolve(null); });
    req.write(body);
    req.end();
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
    console.log(`🔍 Buscando en web: "${query.slice(0, 80)}..."`);
    const searchResults = await doTavilySearch(query);
    if (searchResults) {
      systemPrompt = injectSearchResults(systemPrompt, searchResults);
      console.log("✅ Resultados Tavily inyectados");
    } else {
      const today = new Date().toLocaleDateString("es-ES", {
        weekday: "long", year: "numeric", month: "long", day: "numeric"
      });
      systemPrompt += `\n\n⚠️ Fecha actual: ${today}. NUNCA inventes cifras o estadísticas recientes.`;
      console.log("⚠️  Sin resultados web");
    }
  }

  if (PROVIDER === "groq") {
    // ── Selección automática de modelo ──────────────────────────────────
    // Si el mensaje tiene imagen → llama-4-scout (visión)
    // Si es solo texto          → llama-3.3-70b (razonamiento)
    const hasImage = messageHasImage(rawMessages);
    const model = hasImage ? GROQ_VISION_MODEL : GROQ_TEXT_MODEL;
    const useVision = hasImage;

    console.log(`\n→ GROQ [${model}] imagen:${hasImage} msgs:${rawMessages.length}`);

    const msgs = flattenForOpenAI(rawMessages, systemPrompt, useVision);
    if (!msgs) throw new Error("No hay mensajes válidos.");

    body = JSON.stringify({ model, messages: msgs, max_tokens: 4000, temperature: 0.3 });
    hostname = "api.groq.com";
    urlPath  = "/openai/v1/chat/completions";
    headers  = {
      "Content-Type":   "application/json",
      "Authorization":  `Bearer ${process.env.GROQ_API_KEY}`,
      "Content-Length": Buffer.byteLength(body),
    };
    console.log(`   body: ${body.length}b`);

  } else if (PROVIDER === "anthropic") {
    const model = process.env.ANTHROPIC_MODEL || "claude-sonnet-4-20250514";
    const msgs = flattenForAnthropic(rawMessages);
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
    const model = process.env.OPENAI_MODEL || "gpt-4o";
    const msgs = flattenForOpenAI(rawMessages, systemPrompt, true);
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
  if (PROVIDER === "anthropic") {
    if (status !== 200) {
      try {
        const e = JSON.parse(raw);
        const msg = e.error?.message || raw.slice(0, 200);
        return JSON.stringify({ content: [{ type: "text", text: "Error API: " + msg }] });
      } catch { return raw; }
    }
    try {
      const data = JSON.parse(raw);
      const textBlocks = (data.content || []).filter(b => b.type === "text");
      if (textBlocks.length > 0) return JSON.stringify({ content: textBlocks });
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
  console.log("\n╔══════════════════════════════════════════════╗");
  console.log("║         🤖  KENYRA IA  🤖                    ║");
  console.log("╠══════════════════════════════════════════════╣");
  console.log(`║  Puerto:         ${PORT}                          ║`);
  console.log(`║  Provider:       ${PROVIDER.toUpperCase().padEnd(28)}║`);
  console.log(`║  Modelo texto:   ${GROQ_TEXT_MODEL.padEnd(28)}║`);
  console.log(`║  Modelo visión:  ${GROQ_VISION_MODEL.slice(0,28).padEnd(28)}║`);
  console.log(`║  Web Search:     ${process.env.TAVILY_API_KEY ? "✅ Tavily activo              " : "⚠️  Sin TAVILY_API_KEY        "}║`);
  console.log("╚══════════════════════════════════════════════╝\n");
});

server.on("error", err => {
  if (err.code === "EADDRINUSE") console.error(`❌ Puerto ${PORT} en uso.`);
  else console.error("❌ Error:", err.message);
  process.exit(1);
});
