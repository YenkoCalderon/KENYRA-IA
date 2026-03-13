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

// Convert Anthropic-style content to OpenAI vision format
// Anthropic: {type:"image", source:{type:"base64", media_type:"image/png", data:"..."}}
// OpenAI:    {type:"image_url", image_url:{url:"data:image/png;base64,..."}}
function toOpenAIContent(content) {
  if (!content) return null;

  // Plain string
  if (typeof content === "string") {
    return content.trim() || null;
  }

  // Array of blocks
  if (Array.isArray(content)) {
    const parts = [];
    for (const block of content) {
      if (!block || !block.type) continue;
      if (block.type === "text") {
        const t = (block.text || "").trim();
        if (t) parts.push({ type: "text", text: t });
      } else if (block.type === "image") {
        // Convert Anthropic image block → OpenAI image_url block
        const src = block.source || {};
        if (src.type === "base64" && src.data) {
          const mime = src.media_type || "image/png";
          parts.push({
            type: "image_url",
            image_url: { url: `data:${mime};base64,${src.data}` }
          });
        } else if (src.type === "url" && src.url) {
          parts.push({ type: "image_url", image_url: { url: src.url } });
        }
      }
    }
    if (parts.length === 0) return null;
    // If only one text part, return as plain string (simpler)
    if (parts.length === 1 && parts[0].type === "text") return parts[0].text;
    return parts;
  }

  return String(content).trim() || null;
}

// Flatten to plain string (for history messages that had images — strip them)
function contentToString(content) {
  if (!content) return null;
  if (typeof content === "string") return content.trim() || null;
  if (Array.isArray(content)) {
    // For past history messages: extract text only, replace images with placeholder
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

// Build Groq/OpenAI messages array
// - History messages: plain strings (no images to save tokens)
// - Current (last) user message: full vision content with images
function flattenForOpenAI(messages, systemPrompt) {
  const result = [];
  if (systemPrompt) result.push({ role: "system", content: systemPrompt });

  let lastRole = "system";

  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    const role = m.role === "assistant" ? "assistant" : "user";
    const isLast = (i === messages.length - 1);

    let content;
    if (isLast && role === "user") {
      // Last user message: keep images in OpenAI vision format
      content = toOpenAIContent(m.content);
    } else {
      // History: plain strings only
      content = contentToString(m.content);
    }

    if (!content) continue;

    if (role === lastRole) {
      // Merge same-role consecutive messages
      const prev = result[result.length - 1];
      if (typeof prev.content === "string" && typeof content === "string") {
        prev.content += "\n" + content;
      } else {
        // Convert both to arrays and merge
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

// For Anthropic: keep native image blocks, ensure alternation
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
      if (content.every(b => b.type === "text")) {
        content = content.map(b => b.text).join("\n");
      }
    } else {
      content = String(content).trim();
      if (!content) continue;
    }

    if (role === lastRole) {
      const prev = result[result.length - 1];
      const toStr = c => typeof c === "string" ? c
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

function callAPI(payload) {
  return new Promise((resolve, reject) => {
    const rawMessages = payload.messages || [];
    const systemPrompt = payload.system || "";
    let hostname, urlPath, headers, body;

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

      const apiPayload = { model, max_tokens: 1500, messages: msgs };
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
      console.log(`\n→ ANTHROPIC [${model}] msgs:${msgs.length} body:${body.length}b`);

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
    return raw;
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
  console.log("╚══════════════════════════════════════╝\n");
});

server.on("error", err => {
  if (err.code === "EADDRINUSE") console.error(`❌ Puerto ${PORT} en uso.`);
  else console.error("❌ Error:", err.message);
  process.exit(1);
});