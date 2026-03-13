# 🤖 Kenyra IA

Asistente de inteligencia artificial conversacional con soporte de **visión de imágenes**, historial persistente, descarga de archivos y más.

---

## 📁 Estructura del proyecto

```
kenyra-ia/
├── index.html       ← Interfaz principal (todo en un solo archivo)
├── server.js        ← Proxy Node.js (lee .env, llama a la API de IA)
├── login.html       ← Pantalla de login con Google (Firebase)
├── KENYRAIA.PNG     ← Logo de Kenyra IA
├── .env             ← Tu configuración (NO subir a Git)
├── package.json     ← Metadata del proyecto
└── README.md
```

---

## 🚀 Instalación y uso

### 1. Asegúrate de tener Node.js instalado

```bash
node -v   # debe mostrar v14 o superior
```

Si no lo tienes: https://nodejs.org

### 2. Configura el archivo `.env`

```env
AI_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
PORT=3000
```

> `llama-4-scout` es gratuito en Groq y soporta visión de imágenes.

### 3. Inicia el servidor

```bash
node server.js
```

### 4. Abre en el navegador

```
http://localhost:3000
```

---

## 🌐 Deploy en Railway (URL pública)

1. Sube el proyecto a GitHub
2. Entra a https://railway.app y conecta tu repo
3. Agrega las variables de entorno en Railway (las mismas del `.env`)
4. Railway detecta `package.json` y ejecuta `node server.js` automáticamente

---

## ✨ Funcionalidades

| Función | Descripción |
|---|---|
| 💬 Chat conversacional | Responde preguntas, genera código, tablas, SQL y más |
| 🖼️ Visión de imágenes | Adjunta o pega imágenes (Ctrl+V) y Kenyra las analiza |
| 📎 Adjuntar archivos | Soporta .js, .py, .html, .csv, .sql, .json y más |
| 🗂️ Drag & Drop | Arrastra archivos directamente al chat |
| 📄 Descargar archivos | Genera y descarga código, SQL, PDFs desde el chat |
| 🌅 Saludo dinámico | Buenos días / tardes / noches según la hora |
| 💾 Historial persistente | Los chats se guardan en el navegador (localStorage) |
| 🗑️ Eliminar chats | Botón de papelera en cada conversación |
| 🌙 Tema oscuro / claro | Cambia entre oscuro, claro o sistema |
| 📱 Responsive | Funciona en móvil y desktop |
| ☰ Menú móvil | Drawer lateral para historial en celular |
| 🔐 Login con Google | Autenticación via Firebase |

---

## 🤖 Proveedores de IA compatibles

| Provider | Gratis | Ve imágenes | Modelo recomendado |
|---|---|---|---|
| **Groq** ✅ (activo) | Sí, 14,400 req/día | ✅ Sí | `meta-llama/llama-4-scout-17b-16e-instruct` |
| Groq (solo texto) | Sí | ❌ No | `llama-3.3-70b-versatile` |
| Anthropic (Claude) | No | ✅ Sí | `claude-sonnet-4-20250514` |
| OpenAI | No | ✅ Sí | `gpt-4o` |

### Cambiar de provider

En `.env` cambia `AI_PROVIDER` y el modelo correspondiente:

```env
# Groq con visión (recomendado - gratis)
AI_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct

# Anthropic
AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514

# OpenAI
AI_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
```

---

## 🔒 Seguridad

- La API key **nunca** se expone al navegador
- El servidor Node.js actúa como proxy seguro
- El archivo `.env` **no debe subirse** a repositorios públicos — agrégalo a `.gitignore`

---

## 📋 Requisitos

- Node.js 14 o superior
- Sin dependencias npm adicionales — solo módulos nativos de Node.js
- API key de Groq (gratuita) en https://console.groq.com/keys