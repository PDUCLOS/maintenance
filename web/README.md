# Web — project showcase

A static, self-contained website for the **Industrial Knowledge Copilot** project.
Two pages, no build step, no JS framework — just HTML, CSS, and a tiny bit of
vanilla JS for the demo page.

## Pages

| File | Purpose |
|------|---------|
| `index.html` | Landing / showcase page. Project pitch, architecture, tech stack, A/B test results, RAGAS targets, EU AI Act compliance, quickstart, bugs found. |
| `demo.html`  | Interactive chat interface that talks to the FastAPI backend on `:8000`. Sidebar with live system status + example questions, right panel with retrieved sources. |
| `styles.css`  | Design system: dark mode tokens, glassmorphism, responsive grid, code/table/card components. ~600 lines. |

## How to serve

The site is pure static — no build step. Any HTTP server works.

### Option 1 — Python (simplest, no install)

```bash
cd web
python3 -m http.server 8080
# Open http://localhost:8080
```

### Option 2 — Netlify / Vercel / GitHub Pages

The `web/` directory is the publish root. Push to a `gh-pages` branch or
connect the repo to Netlify for one-click deploy.

### Option 3 — npx serve (if you have Node)

```bash
cd web
npx serve .
```

## Talking to the FastAPI backend

The `demo.html` page fetches from `http://localhost:8000`. CORS is already
configured in the FastAPI app (`src/api/main.py`) to allow the Streamlit
UI on `:8501`. **The web demo at `:8080` is not in the CORS allowlist by
default** — you have two options:

1. **Run a tiny CORS proxy** in front of the API on the same origin
2. **Add the web origin to the CORS allowlist** in `src/api/main.py`:

```python
allow_origins=[
    f"http://localhost:{settings.ui_port}",   # Streamlit
    "http://localhost:8080",                  # Static web demo
],
```

Without one of these, the demo page will show network errors in the
browser console (CORS rejection). The page itself will still render —
the status check just won't be able to reach the API.

## Design choices

- **Dark mode primary** (industry technical audience, no eye strain)
- **Glassmorphism** (subtle backdrop blur, semi-transparent surfaces)
- **System fonts** (no Google Fonts dependency → works fully offline)
- **Responsive** (3-col → 1-col on narrow screens)
- **Accessible** (semantic HTML, focus rings, sufficient contrast)
- **No build step** (edit a file, refresh the browser)

## Editing

The pages are hand-written HTML. To customize:

- **Colors / spacing** → edit tokens at the top of `styles.css`
- **Content** → edit HTML directly in `index.html` or `demo.html`
- **Demo questions** → edit the `EXAMPLES` array at the top of the
  inline `<script>` in `demo.html`
- **New pages** → copy `index.html`, change content, link from header

## Re-generate after project changes

The web pages are a snapshot of the project state. After major milestones
(switching LLM, adding pages, RAGAS results), update the affected HTML
files and re-deploy.

For automatic sync with the markdown docs, see
`scripts/generate_overview_docx.py` (which does the same for the
Word document).
