import express from "express";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";

// --- 1) Your finance API settings come from environment variables ---
const API_BASE = process.env.FIN_API_BASE; // e.g. https://api.yourfinance.com
const API_KEY  = process.env.FIN_API_KEY;  // keep this secret (Render will store it)

// --- 2) Helper that fetches a quote from your API ---
async function getQuote(symbol) {
  const url = `${API_BASE}/v1/quote?symbol=${encodeURIComponent(symbol)}`;
  const res = await fetch(url, { headers: { Authorization: `Bearer ${API_KEY}` } });
  if (!res.ok) throw new Error(`Quote fetch failed: ${res.status}`);
  const data = await res.json();
  return {
    symbol,
    last: data.price ?? data.last ?? null,
    changePct: data.change_pct ?? null,
    currency: data.currency ?? "USD",
    asOf: data.as_of ?? new Date().toISOString()
  };
}

// --- 3) The MCP server describes your tools and UI resources ---
const mcp = new McpServer({ name: "fin-mcp", version: "1.0.0" });

// (A) Register the UI template (inline card) — note mimeType: text/html+skybridge
mcp.registerResource("quote-card", "ui://widget/quote.html", {}, async () => ({
  contents: [{
    uri: "ui://widget/quote.html",
    mimeType: "text/html+skybridge",
    text: `
<div id="root" style="font-family: system-ui, sans-serif; padding:12px"></div>
<script>
  const root = document.getElementById("root");

  function render() {
    const q = (window.openai && (window.openai.widgetState || window.openai.toolOutput)) || null;
    if (!q) { root.textContent = "No data yet."; return; }
    const pct = (q.changePct == null) ? "—" : (q.changePct * 100).toFixed(2) + "%";
    root.innerHTML = `
      <div style="opacity:.7;font-size:14px">${q.symbol}</div>
      <div style="font-size:28px;font-weight:600">${q.last ?? "—"} ${q.currency}</div>
      <div style="font-size:14px">Change: ${pct} • as of ${new Date(q.asOf).toLocaleString()}</div>
      <div style="margin-top:8px">
        <button id="refresh">Refresh</button>
      </div>`;

    // Persist state so it survives re-renders
    window.openai?.setWidgetState?.(q);

    document.getElementById("refresh")?.addEventListener("click", async () => {
      const sym = (window.openai?.toolOutput?.symbol) || "AAPL";
      await window.openai?.callTool?.("get_quote", { symbol: sym });
    });
  }

  window.addEventListener("openai:set_globals", render);
  render();
</script>`.trim(),
    _meta: {
      "openai/widgetDescription": "Real-time stock quote card with a Refresh button",
      "openai/widgetPrefersBorder": true
    }
  }]
}));

// (B) Register the tool that fetches quotes and links to the template above
mcp.registerTool(
  "get_quote",
  {
    title: "Get Quote",
    description: "Fetch latest price/changes for a ticker",
    inputSchema: {
      type: "object",
      properties: { symbol: { type: "string" } },
      required: ["symbol"],
      additionalProperties: false
    },
    _meta: {
      "openai/outputTemplate": "ui://widget/quote.html",
      "openai/widgetAccessible": true,
      "openai/toolInvocation/invoking": "Fetching quote…",
      "openai/toolInvocation/invoked": "Quote updated"
    }
  },
  async ({ symbol }) => {
    const q = await getQuote(symbol);
    return {
      structuredContent: q,
      content: [{ type: "text", text: `Quote for ${q.symbol}: ${q.last ?? "n/a"} ${q.currency}` }]
    };
  }
);

// --- 4) HTTP glue (Streamable HTTP transport at POST /mcp) ---
const app = express();
app.use(express.json());
app.post("/mcp", async (req, res) => {
  const transport = new StreamableHTTPServerTransport({ enableJsonResponse: true });
  res.on("close", () => transport.close());
  await mcp.connect(transport);
  await transport.handleRequest(req, res, req.body);
});

// Health check for Render
app.get("/healthz", (_, res) => res.send("ok"));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`MCP server listening on ${PORT}`));
