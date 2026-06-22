"""The read-only dashboard served at GET /dashboard.

A single self-contained HTML page (no build step, no external assets) that talks
to the existing /v1 endpoints: it lists indexes and runs a text-query search
playground. Embedded as a Python string so it always ships in the wheel.
"""

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Vektoria — dashboard</title>
<style>
  :root { --bg:#05060f; --panel:#0e1120; --border:#1c2236; --fg:#e8ecff;
          --muted:#8b93b8; --cyan:#00e5ff; --purple:#a371f7; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font-family:'Segoe UI',system-ui,sans-serif; }
  header { display:flex; align-items:center; gap:12px; padding:16px 24px;
           border-bottom:1px solid var(--border); }
  header .logo { color:var(--cyan); font-size:22px; }
  header h1 { font-size:18px; margin:0; font-weight:600; letter-spacing:-.02em; }
  header .tag { color:var(--muted); font-size:13px; }
  header input { margin-left:auto; background:var(--panel); border:1px solid var(--border);
                 color:var(--fg); border-radius:8px; padding:8px 12px; font-size:13px; width:240px; }
  main { display:grid; grid-template-columns:280px 1fr; gap:20px; padding:20px 24px; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:16px; }
  .card h2 { font-size:13px; text-transform:uppercase; letter-spacing:.08em;
             color:var(--muted); margin:0 0 12px; }
  .idx { padding:10px 12px; border-radius:8px; cursor:pointer; border:1px solid transparent; }
  .idx:hover { border-color:var(--border); }
  .idx.sel { border-color:var(--cyan); background:rgba(0,229,255,.06); }
  .idx .name { font-weight:600; }
  .idx .meta { color:var(--muted); font-size:12px; }
  .search { display:flex; gap:10px; margin-bottom:16px; }
  .search input { flex:1; background:var(--bg); border:1px solid var(--border); color:var(--fg);
                  border-radius:8px; padding:11px 14px; font-size:14px; }
  .search button { background:linear-gradient(135deg,var(--cyan),var(--purple)); color:#05060f;
                   border:none; border-radius:8px; padding:0 20px; font-weight:600; cursor:pointer; }
  .hit { border:1px solid var(--border); border-radius:10px; padding:14px; margin-bottom:10px; }
  .hit .top { display:flex; justify-content:space-between; color:var(--muted); font-size:12px; }
  .hit .score { color:var(--cyan); font-weight:600; }
  .hit .text { margin:8px 0 0; }
  .hit .src { color:var(--muted); font-size:12px; margin-top:6px; }
  .empty, .err { color:var(--muted); font-size:14px; }
  .err { color:#ff6b8a; }
  code { color:var(--cyan); }
</style>
</head>
<body>
  <header>
    <span class="logo">⬢</span>
    <h1>Vektoria</h1>
    <span class="tag">read-only dashboard</span>
    <input id="apikey" type="password" placeholder="API key (if required)" />
  </header>
  <main>
    <section class="card">
      <h2>Indexes</h2>
      <div id="indexes"><span class="empty">loading…</span></div>
    </section>
    <section class="card">
      <h2>Search playground</h2>
      <div class="search">
        <input id="q" placeholder="Ask a question (the server embeds it)…" />
        <button onclick="runSearch()">Search</button>
      </div>
      <div id="results"><span class="empty">Pick an index, then search.</span></div>
    </section>
  </main>
<script>
  let selected = null;
  const $ = id => document.getElementById(id);

  // Index data (ids, sources, chunk text) is attacker-supplied via upsert/ingest,
  // so it's rendered through textContent / DOM nodes — never innerHTML — and can't
  // inject markup. The same goes for server error messages, which echo ids back.
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };
  const note = (target, cls, text) => $(target).replaceChildren(el("span", cls, text));

  const headers = () => {
    const h = { "Content-Type": "application/json" };
    const k = $("apikey").value.trim();
    if (k) h["Authorization"] = "Bearer " + k;
    return h;
  };

  function renderIndex(ix) {
    const d = el("div", "idx");
    d.append(el("div", "name", ix.name),
             el("div", "meta", `${ix.count} vectors · dim ${ix.dimension} · ${ix.metric}`));
    d.onclick = () => {
      selected = ix.name;
      document.querySelectorAll(".idx").forEach(x => x.classList.remove("sel"));
      d.classList.add("sel");
    };
    return d;
  }

  function renderHit(m) {
    const top = el("div", "top");
    top.append(el("span", null, m.id), el("span", "score", m.score.toFixed(3)));
    const hit = el("div", "hit");
    hit.append(top, el("div", "text", (m.metadata.text || "").slice(0, 400)));
    if (m.metadata.source) hit.append(el("div", "src", "source: " + m.metadata.source));
    return hit;
  }

  async function loadIndexes() {
    try {
      const r = await fetch("v1/indexes", { headers: headers() });
      if (!r.ok) throw new Error("HTTP " + r.status);
      const { indexes } = await r.json();
      if (!indexes.length) return note("indexes", "empty", "No indexes yet.");
      $("indexes").replaceChildren(...indexes.map(renderIndex));
    } catch (e) { note("indexes", "err", e.message); }
  }

  async function runSearch() {
    if (!selected) return note("results", "err", "Select an index first.");
    const q = $("q").value.trim();
    if (!q) return;
    note("results", "empty", "searching…");
    try {
      const r = await fetch("v1/indexes/" + encodeURIComponent(selected) + "/query",
        { method: "POST", headers: headers(), body: JSON.stringify({ text: q, top_k: 10 }) });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail || ("HTTP " + r.status));
      const { matches } = body;
      if (!matches.length) return note("results", "empty", "No matches.");
      $("results").replaceChildren(...matches.map(renderHit));
    } catch (e) { note("results", "err", e.message); }
  }

  $("q").addEventListener("keydown", e => { if (e.key === "Enter") runSearch(); });
  loadIndexes();
</script>
</body>
</html>"""
