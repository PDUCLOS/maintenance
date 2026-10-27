"""Generate docs/architecture.drawio (6 pages) from a compact description.

Writing the diagrams as code keeps them reviewable in a diff and easy to
re-layout. Export to PNG (one file per page) with the draw.io CLI:

    draw.io -x -f png -s 2 -p <page index> -o docs/img/architecture-<n>.png docs/architecture.drawio

Usage: python scripts/make_architecture_drawio.py
"""

from __future__ import annotations

from html import escape
from itertools import pairwise
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "architecture.drawio"

BLUE = "fillColor=#dae8fc;strokeColor=#6c8ebf;"
GREEN = "fillColor=#d5e8d4;strokeColor=#82b366;"
ORANGE = "fillColor=#ffe6cc;strokeColor=#d79b00;"
PURPLE = "fillColor=#e1d5e7;strokeColor=#9673a6;"
YELLOW = "fillColor=#fff2cc;strokeColor=#d6b656;"
GREY = "fillColor=#f5f5f5;strokeColor=#666666;"
WHITE = "fillColor=#ffffff;strokeColor=#666666;"
RED = "fillColor=#f8cecc;strokeColor=#b85450;"


class Page:
    def __init__(self, pid: str, name: str, title: str, subtitle: str, w: int = 1500, h: int = 900):
        self.pid, self.name, self.w, self.h = pid, name, w, h
        self.cells: list[str] = []
        self.n = 0
        self.text(title, 40, 15, w - 80, 30, size=20, bold=True)
        self.text(subtitle, 40, 48, w - 80, 22, size=13, color="#555555")

    def _id(self) -> str:
        self.n += 1
        return f"{self.pid}-{self.n}"

    def text(self, value, x, y, w, h, size=12, bold=False, color="#333333", align="left"):
        cid = self._id()
        style = (
            f"text;html=1;whiteSpace=wrap;fontSize={size};align={align};verticalAlign=middle;"
            f"fontColor={color};" + ("fontStyle=1;" if bold else "")
        )
        self.cells.append(self._vertex(cid, value, style, x, y, w, h))
        return cid

    def group(self, label, x, y, w, h, color=GREY, dashed=False):
        cid = self._id()
        style = (
            "rounded=1;arcSize=3;whiteSpace=wrap;html=1;verticalAlign=top;align=left;"
            "spacingLeft=10;spacingTop=4;fontStyle=1;fontSize=13;"
            + color
            + ("dashed=1;" if dashed else "")
        )
        self.cells.append(self._vertex(cid, label, style, x, y, w, h))
        return cid

    def box(self, label, x, y, w, h, color=WHITE, size=12, bold_border=False, dashed=False):
        cid = self._id()
        style = (
            f"rounded=1;whiteSpace=wrap;html=1;fontSize={size};"
            + color
            + ("strokeWidth=2;" if bold_border else "")
            + ("dashed=1;" if dashed else "")
        )
        self.cells.append(self._vertex(cid, label, style, x, y, w, h))
        return cid

    def db(self, label, x, y, w, h, color=PURPLE, size=12):
        cid = self._id()
        style = (
            f"shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;size=12;fontSize={size};" + color
        )
        self.cells.append(self._vertex(cid, label, style, x, y, w, h))
        return cid

    def step(self, num, x, y):
        cid = self._id()
        style = (
            "ellipse;whiteSpace=wrap;html=1;fontSize=13;fontStyle=1;fontColor=#ffffff;"
            "fillColor=#d79b00;strokeColor=#d79b00;"
        )
        self.cells.append(self._vertex(cid, str(num), style, x, y, 28, 28))
        return cid

    def edge(
        self, src, dst, label="", exit=None, entry=None, dashed=False, color="#555555", points=None
    ):
        """Orthogonal arrow. exit/entry = (x, y) anchor in 0..1 of the box."""
        cid = self._id()
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;html=1;endArrow=block;"
            f"endFill=1;strokeWidth=1.5;strokeColor={color};fontSize=11;fontColor=#333333;"
            "labelBackgroundColor=#ffffff;"
        )
        if exit:
            style += f"exitX={exit[0]};exitY={exit[1]};exitDx=0;exitDy=0;"
        if entry:
            style += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
        if dashed:
            style += "dashed=1;"
        pts = ""
        if points:
            pts = (
                '<Array as="points">'
                + "".join(f'<mxPoint x="{x}" y="{y}" />' for x, y in points)
                + "</Array>"
            )
        self.cells.append(
            f'<mxCell id="{cid}" value="{escape(label)}" style="{style}" edge="1" parent="1" '
            f'source="{src}" target="{dst}"><mxGeometry relative="1" as="geometry">{pts}</mxGeometry></mxCell>'
        )
        return cid

    @staticmethod
    def _vertex(cid, value, style, x, y, w, h):
        return (
            f'<mxCell id="{cid}" value="{escape(value)}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" /></mxCell>'
        )

    def xml(self) -> str:
        body = "\n        ".join(self.cells)
        return f"""  <diagram id="{self.pid}" name="{escape(self.name)}">
    <mxGraphModel dx="1600" dy="1000" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{self.w}" pageHeight="{self.h}" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        {body}
      </root>
    </mxGraphModel>
  </diagram>"""


R, L, T, B = (1, 0.5), (0, 0.5), (0.5, 0), (0.5, 1)


def page_overview() -> Page:
    p = Page(
        "p1",
        "1. Vue d'ensemble",
        "Copilote maintenance — vue d'ensemble",
        "Une question en langage naturel → les passages les plus pertinents des catalogues → une réponse "
        "sourcée, générée par un LLM local. Aucune donnée ne sort de la machine.",
        h=760,
    )
    p.group("Utilisateurs", 40, 90, 260, 560, BLUE)
    tech = p.box(
        "<b>Technicien / ingénieur maintenance</b><br>question en français ou en anglais",
        60,
        130,
        220,
        70,
    )
    ui = p.box(
        "<b>Interface Streamlit</b> :8501<br>Chat · Inventaire · RAGAS · Index", 60, 260, 220, 70
    )
    web = p.box("<b>Démo web statique</b><br>web/demo.html (?api=…)", 60, 390, 220, 60)
    p.box(
        "<b>Banc LLMOps</b><br>scripts/llmops_bench.py<br>appelle la chaîne directement",
        60,
        520,
        220,
        60,
        ORANGE,
        bold_border=True,
    )

    p.group("API FastAPI (src/api)", 340, 90, 260, 560, GREEN)
    q = p.box("<b>POST /query</b> (+ /v1/query)<br>délai max 60 s → 504", 360, 130, 220, 60)
    h = p.box("<b>GET /health</b><br>Chroma + LLM prêts ?", 360, 230, 220, 55)
    m = p.box("<b>GET /metrics</b><br>format Prometheus", 360, 320, 220, 55)
    o = p.box("<b>/ingest · /eval · /index</b><br>ingestion, RAGAS, inventaire", 360, 410, 220, 60)

    p.group("Chaîne RAG (src/rag, LangChain LCEL)", 640, 90, 300, 560, YELLOW)
    s1 = p.box("Détection de la langue (FR / EN)", 660, 130, 260, 45)
    s2 = p.box("<b>Recherche hybride</b><br>dense + BM25, fusion RRF", 660, 215, 260, 55)
    s3 = p.box("<b>Reranking</b> cross-encoder<br>15 candidats → 5 passages", 660, 310, 260, 55)
    s4 = p.box("Prompt « réponse miroir »<br>règles : citer, ne rien inventer", 660, 405, 260, 55)
    s5 = p.box("<b>LLM</b> → réponse + sources<br>+ tokens + latence", 660, 500, 260, 55, ORANGE)

    p.group("Modèles et données (100 % local)", 980, 90, 480, 560, PURPLE)
    chroma = p.db("<b>ChromaDB</b><br>4 498 vecteurs bge-m3 (1024 d)", 1000, 130, 210, 80)
    bm25 = p.db("<b>Index BM25</b><br>cache disque (SHA-256)", 1230, 130, 210, 80)
    ce = p.box("<b>ms-marco-MiniLM-L-6</b><br>cross-encoder 22 M param.", 1000, 300, 210, 70, WHITE)
    p.box(
        "<b>Embeddings bge-m3</b> (ingestion + question)<br>Mac : sentence-transformers (MPS)<br>PC : Ollama (GPU)",
        1230,
        290,
        210,
        90,
        WHITE,
        dashed=True,
    )
    llm = p.box(
        "<b>LLM local</b> — choisi par LLM_PROVIDER<br>mlx : Qwen2.5-7B 4 bits (Mac)<br>ollama : qwen3.5:9b (GPU NVIDIA)",
        1000,
        470,
        440,
        90,
        ORANGE,
        bold_border=True,
    )

    p.edge(tech, ui, exit=B, entry=T)
    p.edge(ui, q, "HTTP", exit=R, entry=L)
    p.edge(web, q, exit=R, entry=(0, 0.8))
    p.edge(q, s1, exit=R, entry=L)
    p.edge(s1, s2, exit=B, entry=T)
    p.edge(s2, s3, exit=B, entry=T)
    p.edge(s3, s4, exit=B, entry=T)
    p.edge(s4, s5, exit=B, entry=T)
    p.edge(s2, chroma, "dense", exit=(1, 0.3), entry=L)
    p.edge(s2, bm25, "lexical", exit=(1, 0.7), entry=(0.5, 1))
    p.edge(s3, ce, exit=R, entry=L)
    p.edge(s5, llm, exit=R, entry=L)
    p.edge(s5, m, "compteurs", exit=(0, 0.8), entry=(1, 0.8), dashed=True)
    p.text(
        "Flèches pleines : appel synchrone · pointillés : mesure / métriques",
        40,
        670,
        900,
        24,
        size=12,
        color="#666666",
    )
    _ = (h, o)
    return p


def page_ingestion() -> Page:
    p = Page(
        "p2",
        "2. Ingestion des catalogues",
        "Ingestion : des PDF aux vecteurs (python -m src.ingestion.pipeline)",
        "Mesuré sur ce PC : 2 catalogues SKF publics, 1 734 pages → 4 498 chunks, 247,6 s de bout en bout "
        "(embeddings bge-m3 sur GPU via Ollama ≈ 2 min 23 s).",
        h=620,
    )
    pdf = p.box(
        "<b>Catalogues PDF</b><br>data/raw/pdf/<br>(non versionnés : licence fabricant)",
        40,
        120,
        220,
        90,
        BLUE,
    )
    load = p.box(
        "<b>PyMuPDF</b><br>texte page par page<br>(n° de page conservé)", 320, 120, 200, 90
    )
    chunk = p.box(
        "<b>Découpage récursif</b><br>500 tokens, chevauchement 50<br>paragraphe → ligne → phrase",
        580,
        120,
        230,
        90,
    )
    emb = p.box(
        "<b>Embeddings bge-m3</b><br>multilingue FR/EN, 1024 d<br>normalisés L2",
        870,
        120,
        210,
        90,
        ORANGE,
    )
    chroma = p.db(
        "<b>ChromaDB</b><br>collection bearings_kb<br>(serveur ou disque local)",
        1140,
        110,
        220,
        110,
    )
    jsonl = p.db(
        "<b>chunks.jsonl</b><br>trace d'audit,<br>ré-embedding sans re-parser",
        580,
        330,
        230,
        100,
        GREY,
    )
    bm25 = p.db(
        "<b>Index BM25</b><br>construit au 1er appel,<br>mis en cache (hash du corpus)",
        1140,
        330,
        220,
        100,
        GREY,
    )
    ids = p.box(
        "Identifiant stable par chunk :<br><code>pdf:FICHIER.pdf:p42:0</code><br>→ ré-ingestion idempotente (upsert)",
        870,
        330,
        210,
        100,
        YELLOW,
    )

    for a, b in ((pdf, load), (load, chunk), (chunk, emb), (emb, chroma)):
        p.edge(a, b, exit=R, entry=L)
    p.edge(chunk, jsonl, exit=B, entry=T)
    p.edge(chroma, bm25, "lecture du corpus", exit=B, entry=T, dashed=True)
    p.edge(chunk, ids, exit=(0.8, 1), entry=(0, 0.5), dashed=True)
    p.text(
        "Pourquoi page par page ? Chaque réponse peut citer « fichier + page » : le technicien rouvre le PDF "
        "au bon endroit pour voir le schéma ou le tableau d'origine.",
        40,
        480,
        1300,
        40,
        size=13,
        color="#555555",
    )
    return p


def page_query() -> Page:
    p = Page(
        "p3",
        "3. Une question pas à pas",
        "Une question pas à pas",
        "Exemple : « What shaft seat tolerance class is recommended for rotating inner ring loads? » — "
        "chaque étape est mesurée par le banc.",
        h=640,
    )
    xs = [40, 290, 540, 790, 1040, 1290]
    labels = [
        "<b>Question</b><br>langue détectée<br>(heuristique, sans dépendance)",
        "<b>Dense</b> : bge-m3<br>15 voisins dans Chroma<br>(sens, synonymes, FR↔EN)",
        "<b>Fusion RRF</b><br>score = Σ 1 / (60 + rang)<br>pas de calibrage des scores",
        "<b>Reranking</b><br>cross-encoder relit<br>(question, passage) → top 5",
        "<b>Prompt</b><br>5 passages numérotés<br>+ consignes strictes",
        "<b>LLM local</b><br>réponse dans la langue<br>de la question + sources",
    ]
    ids = []
    for i, (x, lab) in enumerate(zip(xs, labels, strict=True)):
        color = ORANGE if i == 5 else (YELLOW if i in (2, 3) else WHITE)
        ids.append(p.box(lab, x, 130, 200, 100, color))
        p.step(i + 1, x - 10, 118)
    bm = p.box(
        "<b>BM25</b> : mots exacts<br>15 meilleurs passages<br>(désignations : NU 311, SNL…)",
        290,
        300,
        200,
        100,
    )
    for a, b in pairwise(ids):
        p.edge(a, b, exit=R, entry=L)
    p.edge(ids[0], bm, exit=(0.5, 1), entry=L)
    p.edge(bm, ids[2], exit=R, entry=(0.5, 1))
    out = p.box(
        "<b>Réponse API</b><br>answer · sources (fichier, page, score)<br>language · usage (tokens) · latency_ms",
        790,
        300,
        700,
        80,
        GREEN,
    )
    p.edge(ids[5], out, exit=B, entry=(0.85, 0))
    p.text(
        "Garde-fous : hors corpus → « I don't know from the available data » ; aucune valeur numérique "
        "inventée ; vocabulaire métier (jamais « chunk », « embedding »…) ; délai maximal 60 s côté API.",
        40,
        450,
        1450,
        44,
        size=13,
        color="#555555",
    )
    return p


def page_backends() -> Page:
    p = Page(
        "p4",
        "4. Backends interchangeables",
        "Même code, deux machines : le backend se choisit par variable d'environnement",
        "MLX n'existe que sur Apple Silicon. Le provider Ollama ajouté ici fait tourner la même chaîne "
        "sous Windows / Linux, sans toucher au chemin Mac d'origine.",
        h=660,
    )
    p.group("Mac Apple Silicon (configuration d'origine)", 40, 100, 420, 400, BLUE)
    p.box("LLM_PROVIDER=<b>mlx</b><br>Qwen2.5-7B-Instruct 4 bits (Metal)", 60, 140, 380, 60)
    p.box("EMBED_PROVIDER=<b>sentence_transformers</b><br>bge-m3 sur MPS", 60, 220, 380, 60)
    p.box("TORCH_DEVICE=<b>mps</b> (reranker)", 60, 300, 380, 50)
    p.box("CHROMA_MODE=<b>http</b><br>serveur Chroma dans Docker", 60, 370, 380, 60)
    p.text(
        "Non ré-exécuté dans cette session (pas de Mac).",
        60,
        450,
        380,
        30,
        size=12,
        color="#b85450",
    )

    core = p.group("Code commun (inchangé)", 540, 100, 420, 400, YELLOW)
    c1 = p.box("<b>get_chat_model()</b><br>src/rag/llm.py", 560, 150, 380, 60, WHITE)
    c2 = p.box("<b>Embedder</b> · <b>VectorStore</b> · <b>Reranker</b>", 560, 240, 380, 50, WHITE)
    c3 = p.box("<b>RAGChain</b> · <b>HybridRetriever</b> · API · UI", 560, 320, 380, 50, WHITE)
    p.box(
        "settings.assert_backend_supported()<br>refuse MLX hors Apple Silicon (échec explicite)",
        560,
        400,
        380,
        60,
        RED,
    )

    p.group("PC Windows 11 + RTX 4070 Ti SUPER (cette session)", 1040, 100, 420, 400, GREEN)
    w1 = p.box("LLM_PROVIDER=<b>ollama</b><br>qwen3.5:9b (GPU, think=false)", 1060, 140, 380, 60)
    w2 = p.box("EMBED_PROVIDER=<b>ollama</b><br>bge-m3 via /api/embed", 1060, 220, 380, 60)
    p.box("TORCH_DEVICE=<b>auto</b> → cpu ici (torch CPU)", 1060, 300, 380, 50)
    w4 = p.box(
        "CHROMA_MODE=<b>persistent</b><br>base embarquée sur disque, sans Docker",
        1060,
        370,
        380,
        60,
    )
    p.text("Testé ici : suite pytest + banc complet.", 1060, 450, 380, 30, size=12, color="#2d7a2d")
    p.edge(w1, c1, exit=L, entry=R)
    p.edge(w2, c2, exit=L, entry=R)
    p.edge(w4, c3, exit=L, entry=(1, 0.5))
    p.text(
        "Correctifs de reproductibilité trouvés en installant à neuf : pyarrow et anyio non épinglés "
        "(incompatibles avec numpy 1.26 / starlette 0.41), chroma-hnswlib sans roue Windows.",
        40,
        540,
        1420,
        44,
        size=13,
        color="#555555",
    )
    _ = core
    return p


def page_bench() -> Page:
    p = Page(
        "p5",
        "5. Banc LLMOps",
        "Banc LLMOps : comment les chiffres de la vitrine sont produits",
        "scripts/build_gold_set.py puis scripts/llmops_bench.py — résultats bruts dans reports/llmops/latest.json.",
        h=700,
    )
    pages = p.box(
        "<b>Pages de catalogue</b><br>riches en texte, tirées<br>au sort (graine 42)",
        40,
        130,
        200,
        90,
        BLUE,
    )
    gen = p.box(
        "<b>gemma4:e4b</b> écrit 1 question<br>précise par page<br>(1 sur 3 en français)",
        300,
        130,
        220,
        90,
        PURPLE,
    )
    gold = p.db("<b>30 questions</b><br>« page connue »<br>data/eval/", 580, 125, 180, 100, GREY)
    proj = p.db(
        "<b>25 questions</b><br>jeu d'origine<br>(src/eval/dataset.py)", 330, 300, 180, 100, GREY
    )
    p.edge(pages, gen, exit=R, entry=L)
    p.edge(gen, gold, exit=R, entry=L)

    p.group("Recherche (sans LLM)", 830, 100, 300, 300, YELLOW)
    r = p.box("dense · BM25 · hybride RRF<br>· hybride + reranking", 850, 140, 260, 60)
    rm = p.box("hit@1 · hit@5 · MRR@5<br>page ±1 · FR vs EN · latence", 850, 230, 260, 60, WHITE)
    p.edge(r, rm, exit=B, entry=T)
    p.edge(gold, r, exit=R, entry=L)
    p.edge(proj, r, exit=R, entry=(0, 0.8), points=[(800, 350), (800, 188)])

    p.group("Génération + juges", 830, 430, 620, 220, ORANGE)
    g = p.box("<b>qwen3.5:9b</b> répond<br>(chaîne de production)", 850, 480, 180, 70)
    j1 = p.box(
        "<b>Juge 1 : gemma4:e4b</b><br>autre famille de modèle<br>fidélité + exactitude 1-5",
        1060,
        470,
        180,
        90,
        PURPLE,
    )
    j2 = p.box(
        "<b>Juge 2 : qwen3.5:9b</b><br>auto-évaluation : sert<br>seulement à l'accord",
        1260,
        470,
        170,
        90,
        WHITE,
    )
    p.edge(g, j1, exit=R, entry=L)
    p.edge(j1, j2, exit=R, entry=L)
    p.box(
        "latence p50/p95 · tokens réels Ollama · tokens/s · refus hors corpus · injection de prompt",
        850,
        580,
        580,
        45,
        WHITE,
        size=11,
    )
    p.edge(gold, g, "30 + 5 hors corpus + 1 injection", exit=B, entry=L)
    rep = p.box(
        "<b>reports/llmops/latest.json</b><br>→ vitrine + PORTFOLIO.md",
        1200,
        170,
        240,
        70,
        GREEN,
        bold_border=True,
    )
    p.edge(rm, rep, exit=R, entry=(0.5, 1))
    p.edge(j2, rep, exit=T, entry=(0.8, 1))
    p.text(
        "Le générateur des questions (gemma) n'est pas le modèle évalué (qwen) : on limite le biais « le modèle se pose ses propres questions ».",
        40,
        610,
        740,
        60,
        size=12,
        color="#555555",
    )
    return p


def page_k8s() -> Page:
    p = Page(
        "p6",
        "6. Déploiement Kubernetes",
        "Déploiement Kubernetes (deploy/k8s, kustomize : base + 2 overlays)",
        "Rendu validé hors ligne avec « kubectl kustomize » (19 objets par overlay). Non appliqué à un cluster "
        "dans cette session.",
        h=780,
    )
    p.group("namespace maintenance-copilot", 40, 90, 1100, 640, GREY)
    ing = p.box(
        "<b>Ingress</b><br>copilote.localhost<br>TLS + oauth2-proxy (cloud)", 70, 140, 240, 70, BLUE
    )
    pvc3 = p.db(
        "<b>PVC copilot-data</b><br>PDF, chunks, cache BM25", 70, 280, 240, 80, GREY, size=11
    )
    cfg = p.box(
        "<b>ConfigMap copilot-config</b><br>LLM_PROVIDER=ollama · CHROMA_MODE=http …<br>aucun secret nécessaire",
        70,
        620,
        340,
        70,
        WHITE,
    )
    ui = p.box("<b>copilot-ui</b> (Deployment)<br>Streamlit :8501", 400, 140, 240, 70, WHITE)
    api = p.box(
        "<b>copilot-api</b> (Deployment + HPA 2→4)<br>FastAPI :8000 · /health · /metrics",
        400,
        280,
        240,
        80,
        GREEN,
        bold_border=True,
    )
    ingest = p.box(
        "<b>Job ingest</b><br>python -m src.ingestion.pipeline<br>(écrit dans Chroma)",
        400,
        430,
        240,
        70,
        YELLOW,
    )
    chroma = p.box(
        "<b>chroma</b> (StatefulSet)<br>chromadb/chroma:0.6.3", 760, 140, 240, 70, PURPLE
    )
    oll = p.box(
        "<b>ollama</b> (Deployment, 1 GPU)<br>qwen3.5:9b + bge-m3",
        760,
        280,
        240,
        80,
        ORANGE,
        bold_border=True,
    )
    pull = p.box("<b>Job pull-models</b><br>ollama pull …", 840, 520, 160, 60, YELLOW)
    pvc1 = p.db("PVC chroma-data", 1030, 145, 90, 60, GREY, size=10)
    pvc2 = p.db("PVC ollama-models", 1030, 290, 90, 60, GREY, size=10)
    net = p.box(
        "<b>NetworkPolicy</b> : Chroma, Ollama ← namespace seulement<br>API ← interface + Prometheus seulement",
        640,
        640,
        480,
        60,
        RED,
    )
    p.edge(ing, ui, exit=R, entry=L)
    p.edge(ui, api, "UI_API_BASE_URL", exit=B, entry=T)
    p.edge(api, chroma, exit=(1, 0.3), entry=L)
    p.edge(api, oll, exit=R, entry=L)
    p.edge(pvc3, api, exit=R, entry=L, dashed=True)
    p.edge(pvc3, ingest, exit=B, entry=L, dashed=True)
    p.edge(ingest, oll, "embeddings", exit=R, entry=(0.3, 1), dashed=True)
    p.edge(pull, oll, exit=T, entry=(0.8, 1), dashed=True)
    p.edge(chroma, pvc1, exit=R, entry=L)
    p.edge(oll, pvc2, exit=R, entry=L)

    p.group("Overlays", 1180, 90, 280, 640, BLUE)
    p.box(
        "<b>local-gpu</b><br>Docker Desktop / k3s<br>1 réplique, volumes RWO,<br>pas de nodeSelector GPU",
        1200,
        140,
        240,
        110,
    )
    p.box(
        "<b>cloud</b> (AKS)<br>registre ACR, Azure Files RWX,<br>3 répliques, TLS cert-manager,<br>auth Entra ID (oauth2-proxy)",
        1200,
        290,
        240,
        120,
    )
    p.box(
        "Le goulot est le GPU : ajouter des pods API ne multiplie pas le débit du LLM.",
        1200,
        450,
        240,
        90,
        YELLOW,
    )
    _ = (cfg, net)
    return p


def main() -> int:
    pages = [
        page_overview(),
        page_ingestion(),
        page_query(),
        page_backends(),
        page_bench(),
        page_k8s(),
    ]
    xml = (
        '<mxfile host="app.diagrams.net" agent="scripts/make_architecture_drawio.py" '
        f'version="24.7.17" type="device" pages="{len(pages)}">\n'
        + "\n".join(pg.xml() for pg in pages)
        + "\n</mxfile>\n"
    )
    OUT.write_text(xml, encoding="utf-8")
    print(f"{len(pages)} pages -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
