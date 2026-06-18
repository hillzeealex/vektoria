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
  const headers = () => {
    const h = { "Content-Type": "application/json" };
    const k = $("apikey").value.trim();
    if (k) h["Authorization"] = "Bearer " + k;
    return h;
  };

  async function loadIndexes() {
    try {
      const r = await fetch("v1/indexes", { headers: headers() });
      if (!r.ok) throw new Error("HTTP " + r.status);
      const { indexes } = await r.json();
      const el = $("indexes");
      if (!indexes.length) { el.innerHTML = '<span class="empty">No indexes yet.</span>'; return; }
      el.innerHTML = "";
      indexes.forEach(ix => {
        const d = document.createElement("div");
        d.className = "idx";
        d.innerHTML = `<div class="name">${ix.name}</div>
          <div class="meta">${ix.count} vectors · dim ${ix.dimension} · ${ix.metric}</div>`;
        d.onclick = () => { selected = ix.name; document.querySelectorAll(".idx").forEach(x => x.classList.remove("sel")); d.classList.add("sel"); };
        el.appendChild(d);
      });
    } catch (e) { $("indexes").innerHTML = '<span class="err">' + e.message + '</span>'; }
  }

  async function runSearch() {
    if (!selected) { $("results").innerHTML = '<span class="err">Select an index first.</span>'; return; }
    const q = $("q").value.trim();
    if (!q) return;
    $("results").innerHTML = '<span class="empty">searching…</span>';
    try {
      const r = await fetch("v1/indexes/" + encodeURIComponent(selected) + "/query",
        { method: "POST", headers: headers(), body: JSON.stringify({ text: q, top_k: 10 }) });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail || ("HTTP " + r.status));
      const { matches } = body;
      if (!matches.length) { $("results").innerHTML = '<span class="empty">No matches.</span>'; return; }
      $("results").innerHTML = matches.map(m => `
        <div class="hit">
          <div class="top"><span>${m.id}</span><span class="score">${m.score.toFixed(3)}</span></div>
          <div class="text">${(m.metadata.text || "").replace(/</g,"&lt;").slice(0,400)}</div>
          ${m.metadata.source ? '<div class="src">source: ' + m.metadata.source + '</div>' : ''}
        </div>`).join("");
    } catch (e) { $("results").innerHTML = '<span class="err">' + e.message + '</span>'; }
  }

  $("q").addEventListener("keydown", e => { if (e.key === "Enter") runSearch(); });
  loadIndexes();
</script>
</body>
</html>"""
