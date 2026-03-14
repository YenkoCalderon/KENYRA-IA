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

console.log("🔧 Variables de entorno detectadas:");
console.log("   PORT      :", process.env.PORT      || "❌ no definida");
console.log("   AI_PROVIDER:", process.env.AI_PROVIDER || "❌ no definida");
console.log("   GROQ_API_KEY:", process.env.GROQ_API_KEY ? "✅ existe" : "❌ no definida");
console.log("   GROQ_MODEL :", process.env.GROQ_MODEL || "❌ no definida");
console.log("   SERP_API_KEY:", process.env.SERP_API_KEY ? `✅ existe (${process.env.SERP_API_KEY.slice(0,6)}...${process.env.SERP_API_KEY.slice(-4)})` : "❌ no definida");

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

// ─── Detecta si el mensaje necesita búsqueda web ─────────────────────────
function needsWebSearch(messages) {
  if (!messages || messages.length === 0) return false;
  const last = messages[messages.length - 1];
  const text = typeof last.content === "string"
    ? last.content
    : (Array.isArray(last.content) ? last.content.filter(b => b.type === "text").map(b => b.text).join(" ") : "");
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

// ─── Detecta si es consulta de Tokusatsu ─────────────────────────────────
function isTokusatsuQuery(text) {
  const lower = text.toLowerCase();
  return ["sentai", "kamen rider", "ultraman", "power rangers", "tokusatsu",
          "gorenger", "zyuranger", "gokaiger", "ryusoulger", "donbrothers",
          "king-ohger", "boonboomger", "gozyuger", "shadowrangers"].some(k => lower.includes(k));
}

// ─── Detecta si necesita lista completa ──────────────────────────────────
function needsFullList(text) {
  const lower = text.toLowerCase();
  return ["todos los", "todas las", "lista completa", "dame todos", "dame todas",
          "enumera", "lista de", "cuántos hay", "cuantos hay",
          "todos hasta", "todas hasta", "hasta el 2026", "hasta 2026"].some(k => lower.includes(k));
}

// ─── Fetch contenido de Wikipedia via API ────────────────────────────────
async function fetchWikipedia(query) {
  return new Promise((resolve) => {
    const searchTerm = encodeURIComponent(query.slice(0, 150));
    const options = {
      hostname: "es.wikipedia.org",
      path: `/w/api.php?action=query&list=search&srsearch=${searchTerm}&format=json&srlimit=1`,
      method: "GET",
      headers: { "Accept": "application/json", "User-Agent": "KenyraIA/1.0" }
    };
    const req = https.request(options, res => {
      const chunks = [];
      res.on("data", c => chunks.push(c));
      res.on("end", () => {
        try {
          const data = JSON.parse(Buffer.concat(chunks).toString("utf8"));
          const results = data.query?.search || [];
          if (!results.length) return resolve(null);
          const pageId = results[0].pageid;
          const contentOptions = {
            hostname: "es.wikipedia.org",
            path: `/w/api.php?action=query&pageids=${pageId}&prop=extracts&exintro=false&explaintext=true&format=json&exchars=8000`,
            method: "GET",
            headers: { "Accept": "application/json", "User-Agent": "KenyraIA/1.0" }
          };
          const req2 = https.request(contentOptions, res2 => {
            const chunks2 = [];
            res2.on("data", c => chunks2.push(c));
            res2.on("end", () => {
              try {
                const data2 = JSON.parse(Buffer.concat(chunks2).toString("utf8"));
                const page = data2.query?.pages?.[pageId];
                const extract = page?.extract || "";
                if (!extract) return resolve(null);
                console.log(`📖 Wikipedia: "${page.title}" (${extract.length} chars)`);
                resolve(`[Wikipedia: ${page.title}]\n${extract.slice(0, 6000)}`);
              } catch(e) {
                console.log(`❌ Wikipedia content error: ${e.message}`);
                resolve(null);
              }
            });
          });
          req2.on("error", () => resolve(null));
          req2.end();
        } catch(e) {
          console.log(`❌ Wikipedia search error: ${e.message}`);
          resolve(null);
        }
      });
    });
    req.on("error", () => resolve(null));
    req.end();
  });
}

// ─── Fetch RangerWiki Fandom via SerpApi ─────────────────────────────────
async function fetchRangerWiki(query) {
  const apiKey = process.env.SERP_API_KEY;
  if (!apiKey) return null;
  return new Promise((resolve) => {
    const q = encodeURIComponent(query.slice(0, 150) + " site:powerrangers.fandom.com OR site:shadowrangers.live");
    const options = {
      hostname: "serpapi.com",
      path: `/search.json?q=${q}&api_key=${apiKey}&hl=es&gl=pe&num=5`,
      method: "GET",
      headers: { "Accept": "application/json" }
    };
    const req = https.request(options, res => {
      const chunks = [];
      res.on("data", c => chunks.push(c));
      res.on("end", () => {
        try {
          const data = JSON.parse(Buffer.concat(chunks).toString("utf8"));
          const results = (data.organic_results || []).slice(0, 5);
          if (!results.length) return resolve(null);
          const summary = results.map((r, i) =>
            `[${i+1}] ${r.title}\n${r.snippet || ""}`
          ).join("\n\n");
          console.log(`🦸 RangerWiki/ShadowRangers: ${results.length} resultados`);
          resolve(summary);
        } catch(e) { resolve(null); }
      });
    });
    req.on("error", () => resolve(null));
    req.end();
  });
}

// ─── Búsqueda web con SerpApi ─────────────────────────────────────────────
async function doWebSearch(query) {
  const apiKey = process.env.SERP_API_KEY;
  if (!apiKey) {
    console.log("❌ SERP_API_KEY no está definida en las variables de entorno");
    return null;
  }

  console.log(`🔑 Usando SERP_API_KEY: ${apiKey.slice(0, 6)}...${apiKey.slice(-4)}`);

  // 1. Si es Tokusatsu → RangerWiki + ShadowRangers primero
  if (isTokusatsuQuery(query)) {
    console.log("🦸 Consulta Tokusatsu — buscando en RangerWiki/ShadowRangers...");
    const fandomResult = await fetchRangerWiki(query);
    if (fandomResult) {
      // También combinar con Wikipedia para listas completas
      if (needsFullList(query)) {
        const wikiResult = await fetchWikipedia(query);
        if (wikiResult) {
          return fandomResult + "\n\n" + wikiResult;
        }
      }
      return fandomResult;
    }
  }

  // 2. Si necesita lista completa → Wikipedia
  if (needsFullList(query)) {
    console.log("📖 Solicitud de lista completa — buscando en Wikipedia...");
    const wikiResult = await fetchWikipedia(query);
    if (wikiResult) {
      console.log("✅ Contenido Wikipedia obtenido");
      return wikiResult;
    }
    console.log("⚠️ Wikipedia no encontró resultados, usando SerpApi...");
  }

  // 3. Búsqueda normal con SerpApi
  return new Promise((resolve) => {
    const q = encodeURIComponent(query.slice(0, 200));
    const options = {
      hostname: "serpapi.com",
      path: `/search.json?q=${q}&api_key=${apiKey}&hl=es&gl=pe&num=5`,
      method: "GET",
      headers: { "Accept": "application/json" }
    };
    const req = https.request(options, res => {
      const chunks = [];
      res.on("data", c => chunks.push(c));
      res.on("end", () => {
        try {
          const raw = Buffer.concat(chunks).toString("utf8");
          console.log(`🔎 SerpApi [${res.statusCode}]: ${raw.slice(0, 300)}`);
          const data = JSON.parse(raw);
          const results = (data.organic_results || []).slice(0, 5);
          if (!results.length) {
            console.log("⚠️ SerpApi no devolvió resultados orgánicos");
            return resolve(null);
          }
          const summary = results.map((r, i) =>
            `[${i+1}] ${r.title}\n${r.snippet || ""}`
          ).join("\n\n");
          resolve(summary);
        } catch(e) {
          console.log(`❌ SerpApi parse error: ${e.message}`);
          resolve(null);
        }
      });
    });
    req.on("error", (e) => {
      console.log(`❌ SerpApi network error: ${e.message}`);
      resolve(null);
    });
    req.end();
  });
}

// ─── Extrae el texto del último mensaje ──────────────────────────────────
function extractLastUserText(messages) {
  if (!messages || messages.length === 0) return "";
  const last = messages[messages.length - 1];
  if (typeof last.content === "string") return last.content;
  if (Array.isArray(last.content)) {
    return last.content.filter(b => b.type === "text").map(b => b.text).join(" ");
  }
  return "";
}

// ─── Inyecta resultados de búsqueda en el system prompt ──────────────────
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
    const searchResults = await doWebSearch(query);
    if (searchResults) {
      systemPrompt = injectSearchResults(systemPrompt, searchResults);
      console.log("✅ Resultados web inyectados en el prompt");
    } else {
      const today = new Date().toLocaleDateString("es-ES", {
        weekday: "long", year: "numeric", month: "long", day: "numeric"
      });
      systemPrompt += `\n\n⚠️ Fecha actual: ${today}. Si el usuario pregunta sobre datos en tiempo real, indica que no tienes acceso a internet. NUNCA inventes cifras o estadísticas recientes.`;
      console.log("⚠️  Sin resultados web — fecha inyectada");
    }
  }

  if (PROVIDER === "groq") {
    const model = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";
    const msgs = flattenForOpenAI(rawMessages, systemPrompt);
    if (!msgs) throw new Error("No hay mensajes válidos.");
    body = JSON.stringify({ model, messages: msgs, max_tokens: 4000, temperature: 0.3 });
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
    const msgs = flattenForOpenAI(rawMessages, systemPrompt);
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
  console.log("\n╔══════════════════════════════════════╗");
  console.log("║       🤖  KENYRA IA  🤖              ║");
  console.log("╠══════════════════════════════════════╣");
  console.log(`║  Puerto:    ${PORT}                      ║`);
  console.log(`║  Provider:  ${PROVIDER.toUpperCase().padEnd(26)}║`);
  console.log(`║  Web Search: ${process.env.SERP_API_KEY ? "✅ SerpApi activo        " : "⚠️  Sin SERP_API_KEY     "}║`);
  console.log("╚══════════════════════════════════════╝\n");
});

server.on("error", err => {
  if (err.code === "EADDRINUSE") console.error(`❌ Puerto ${PORT} en uso.`);
  else console.error("❌ Error:", err.message);
  process.exit(1);
});
