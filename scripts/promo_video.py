"""Promo video (1080p, French narration) built from the project's real screenshots.

Adapted from scripts/promo_video.py of the LLMRAGlangchain project.
Pipeline: HTML scenes -> 1920x1080 PNG (Edge headless) -> French narration
(Windows voice "Microsoft Hortense Desktop", System.Speech) -> ffmpeg clips
(slow zoom + fades) -> concat + synthesised background pad (no licensed audio).

Key figures are read from reports/llmops/latest.json (the bench output), so the
video can never quote a number the bench did not measure.

Usage (Windows): python scripts/promo_video.py
Output: docs/video/copilote-maintenance-promo.mp4 (+ docs/video/poster.png)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
OUT = ROOT / "docs" / "video"
WORK = ROOT / "data" / "promo"
REPORT = ROOT / "reports" / "llmops" / "latest.json"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = os.environ.get("FFPROBE") or shutil.which("ffprobe") or "ffprobe"
FPS = 30
PAD_S = 0.8  # breathing time after each sentence

CSS = """
:root { --bg:#14181d; --ink:#eef0f2; --muted:#a8b0ba; --accent:#f0a032; --good:#6cc08b; }
* { box-sizing:border-box; margin:0; }
html,body { width:1920px; height:1080px; overflow:hidden; background:var(--bg); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,sans-serif; }
.stage { width:1920px; height:1080px; padding:110px 140px; display:grid; align-content:center; gap:34px;
  background: radial-gradient(1100px 700px at 88% 12%, #2b2418 0%, transparent 60%), var(--bg); }
.eyebrow { font:500 26px "IBM Plex Mono",monospace; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); }
h1 { font:700 100px/1.03 "IBM Plex Sans Condensed",sans-serif; letter-spacing:-.015em; max-width:1550px; }
h1 em { font-style:normal; color:var(--accent); }
p.sub { font-size:38px; line-height:1.45; color:var(--muted); max-width:1450px; }
ul { list-style:none; padding:0; display:grid; gap:26px; }
li { font-size:44px; line-height:1.3; padding-left:56px; position:relative; }
li::before { content:""; position:absolute; left:0; top:22px; width:28px; height:4px; background:var(--accent); }
.kpis { display:grid; grid-template-columns:repeat(5,1fr); gap:2px; background:rgba(255,255,255,.14); border-radius:10px; overflow:hidden; }
.kpi { background:var(--bg); padding:40px 30px; display:grid; gap:10px; }
.kpi b { font:700 84px "IBM Plex Sans Condensed",sans-serif; color:#fff; }
.kpi span { font-size:27px; color:var(--muted); line-height:1.3; }
.shotstage { width:1920px; height:1080px; display:grid; grid-template-rows:1fr auto; background:var(--bg); }
.frame { display:grid; place-items:center; padding:48px 60px 26px; overflow:hidden; }
.frame img { max-width:100%; max-height:860px; border-radius:12px; box-shadow:0 30px 80px rgba(0,0,0,.5); border:1px solid rgba(255,255,255,.14); }
.frame.light img { background:#fff; padding:14px; }
.band { padding:26px 60px 40px; display:flex; gap:30px; align-items:baseline; }
.band .tag { font:500 24px "IBM Plex Mono",monospace; color:var(--accent); letter-spacing:.12em; text-transform:uppercase; white-space:nowrap; }
.band .cap { font:500 40px/1.3 "IBM Plex Sans",sans-serif; }
.chips { display:flex; flex-wrap:wrap; gap:16px; }
.chip { font:500 30px "IBM Plex Mono",monospace; border:1px solid rgba(255,255,255,.3); border-radius:6px; padding:10px 18px; }
.url { font:500 40px "IBM Plex Mono",monospace; color:var(--accent); }
"""
FONTS = (
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:'
    'wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap">'
)


def fr(x: float, nd: int = 1) -> str:
    """French decimal formatting: 0.83 -> '0,83'."""
    return f"{x:.{nd}f}".replace(".", ",")


def page(body: str) -> str:
    return f'<!doctype html><html lang="fr"><head><meta charset="utf-8">{FONTS}<style>{CSS}</style></head><body>{body}</body></html>'


def card(eyebrow: str, title: str, sub: str = "", extra: str = "") -> str:
    sub_html = f'<p class="sub">{sub}</p>' if sub else ""
    return page(
        f'<div class="stage"><span class="eyebrow">{eyebrow}</span><h1>{title}</h1>{sub_html}{extra}</div>'
    )


def shot(image: str, tag: str, caption: str, light: bool = False) -> str:
    src = (IMG / image).as_uri()
    cls = "frame light" if light else "frame"
    return page(
        f'<div class="shotstage"><div class="{cls}"><img src="{src}"></div>'
        f'<div class="band"><span class="tag">{escape(tag)}</span><span class="cap">{caption}</span></div></div>'
    )


def scenes(rep: dict) -> list[tuple[str, str, str, float]]:
    ret = rep["retrieval"]
    gen = rep["generation"]
    best = ret["hybrid_rrf_rerank"]
    hit5 = round(best["hit@5"] * 100)
    dense5 = round(ret["dense"]["hit@5"] * 100)
    bm5 = round(ret["bm25"]["hit@5"] * 100)
    faith = gen["faithfulness_gemma_mean"]
    p50 = gen["latency_ms_p50"] / 1000
    kpis = (
        '<div class="kpis">'
        + "".join(
            f'<div class="kpi"><b>{v}</b><span>{t}</span></div>'
            for v, t in [
                (f"{hit5} %", "bonne page dans les 5 sources (30 questions)"),
                (f"{fr(faith, 1)} / 5", "fidélité aux sources, juge gemma4"),
                (f"{fr(p50)} s", "latence médiane, qwen3.5:9b sur RTX 4070 Ti"),
                ("1 734", "pages de catalogues indexées et interrogées"),
                ("0 €", "par question : tout est local"),
            ]
        )
        + "</div>"
    )
    return [
        (
            "01-titre",
            card(
                "Copilote maintenance · RAG local",
                "Les catalogues de roulements, <em>interrogés en langage naturel</em>.",
            ),
            "Mille sept cents pages de catalogues de roulements. Une question en français ou en anglais. "
            "Une réponse sourcée, à la page près, calculée sur votre propre machine.",
            1.04,
        ),
        (
            "02-probleme",
            card(
                "Le besoin",
                "Les questions d'un atelier de maintenance",
                extra="<ul><li>« Quelle tolérance d'arbre pour une bague intérieure tournante ? »</li>"
                "<li>« Jusqu'à quelle température tient un joint en caoutchouc nitrile ? »</li>"
                "<li>« Comment démonter un roulement sur manchon de serrage ? »</li></ul>",
            ),
            "Un technicien cherche une tolérance, une température limite, une procédure de démontage. "
            "La réponse existe, mais elle est perdue dans des catalogues de mille pages.",
            1.03,
        ),
        (
            "03-demo",
            shot(
                "demo-chat.png",
                "Démo réelle",
                "Réponse citée, avec le fichier et la page du catalogue.",
            ),
            "Le copilote retrouve les bons passages, répond dans la langue de la question, et cite le fichier et la page. "
            "Le technicien rouvre le PDF au bon endroit.",
            1.10,
        ),
        (
            "04-refus",
            shot(
                "demo-refus.png", "Garde-fou", "Hors catalogue : le copilote dit qu'il ne sait pas."
            ),
            "Si la réponse n'est pas dans les catalogues, comme un prix ou un numéro de téléphone, il le dit, au lieu d'inventer.",
            1.06,
        ),
        (
            "05-archi",
            shot(
                "architecture-3.png",
                "Fonctionnement",
                "Recherche par le sens + par les mots exacts, fusion, reranking, LLM local.",
                light=True,
            ),
            "La recherche combine le sens, avec des vecteurs multilingues, et les mots exacts, pour les désignations comme NU 311. "
            "Un second modèle relit les candidats, puis le LLM rédige la réponse.",
            1.10,
        ),
        (
            "06-gpu",
            card(
                "Porté sous Windows",
                "Conçu pour Mac, il tourne maintenant <em>sur un GPU NVIDIA</em>.",
                "Le moteur MLX d'Apple est conservé ; un second moteur, Ollama, se choisit par variable d'environnement.",
                '<div class="chips"><span class="chip">RTX 4070 Ti SUPER · 16 Go</span><span class="chip">Ollama</span>'
                '<span class="chip">qwen3.5:9b</span><span class="chip">bge-m3</span><span class="chip">ChromaDB</span>'
                '<span class="chip">LangChain</span></div>',
            ),
            "Le projet était écrit pour les puces Apple. Je l'ai porté sur un PC équipé d'une carte NVIDIA, "
            "sans toucher au chemin d'origine : le moteur se choisit par une variable.",
            1.03,
        ),
        (
            "07-mesures",
            shot(
                "bench-retrieval.png",
                "Mesures",
                "Mêmes questions, quatre façons de chercher.",
                light=True,
            ),
            f"Le banc pose trente questions dont on connaît la page. Recherche par le sens seul : {dense5} pour cent. "
            f"Par les mots seuls : {bm5} pour cent. Combinées puis relues : {hit5} pour cent.",
            1.08,
        ),
        (
            "08-chiffres",
            card("Résultats mesurés", "Les chiffres, pas les promesses.", extra=kpis),
            f"Bonne page trouvée dans {hit5} pour cent des cas. Fidélité aux sources : {fr(faith, 1)} sur 5 selon un modèle juge indépendant. "
            f"Environ {fr(p50)} secondes par réponse, et zéro euro par question.",
            1.03,
        ),
        (
            "09-k8s",
            shot(
                "architecture-6.png",
                "Déploiement",
                "Image Docker · manifestes Kubernetes pour un poste GPU ou le cloud.",
                light=True,
            ),
            "Pour aller plus loin : une image Docker et des manifestes Kubernetes, pour un poste GPU ou pour le cloud.",
            1.12,
        ),
        (
            "10-fin",
            card(
                "Copilote maintenance",
                "Le code, la démarche et les mesures sont publics.",
                extra='<p class="url">github.com/PDUCLOS/maintenance</p><p class="sub">Patrice Duclos · Analytics Engineer · Architecte IA</p>',
            ),
            "Copilote maintenance. Le code, la démarche et les mesures sont sur GitHub.",
            1.04,
        ),
    ]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def render_png(html: str, png: Path) -> None:
    src = png.with_suffix(".html")
    src.write_text(html, encoding="utf-8")
    run(
        [
            EDGE,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            f"--user-data-dir={WORK / 'edge-profile'}",
            "--window-size=1920,1080",
            "--virtual-time-budget=4000",
            f"--screenshot={png}",
            src.as_uri(),
        ]
    )


def narrate(text: str, wav: Path) -> None:
    txt = wav.with_suffix(".txt")
    txt.write_text(text, encoding="utf-8")
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$s.SelectVoice('Microsoft Hortense Desktop'); $s.Rate = 1;"
        f"$s.SetOutputToWaveFile('{wav}');"
        f"$s.Speak([IO.File]::ReadAllText('{txt}', [Text.Encoding]::UTF8)); $s.Dispose()"
    )
    run(["powershell", "-NoProfile", "-Command", ps])


def duration(path: Path) -> float:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def encoder() -> list[str]:
    # Slides with a slow zoom compress very well with x264 tuned for still
    # images; the bitrate cap keeps the ~110 s video under the 15 MB budget of
    # GitHub Pages hosting (the constant-quality NVENC first try gave 52 MB).
    return [
        "-c:v", "libx264", "-preset", "slow", "-tune", "stillimage", "-crf", "24",
        "-maxrate", "850k", "-bufsize", "1700k",
    ]  # fmt: skip


def clip(png: Path, wav: Path, zoom: float, mp4: Path) -> float:
    dur = duration(wav) + PAD_S
    frames = int(dur * FPS)
    step = (zoom - 1.0) / frames
    vf = (
        f"[0:v]scale=3840:2160,zoompan=z='min(1+{step:.6f}*on,{zoom})':d={frames}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps={FPS},"
        f"fade=t=in:st=0:d=0.45,fade=t=out:st={dur - 0.45:.2f}:d=0.45,format=yuv420p[v];"
        f"[1:a]adelay=350|350,apad,atrim=0:{dur:.2f},aformat=sample_rates=48000:channel_layouts=stereo[a]"
    )
    run(
        [
            FFMPEG,
            "-y",
            "-loop",
            "1",
            "-i",
            str(png),
            "-i",
            str(wav),
            "-filter_complex",
            vf,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-t",
            f"{dur:.2f}",
            *encoder(),
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            str(mp4),
        ]
    )
    return dur


def main() -> int:
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    clips, total = [], 0.0
    for name, html, text, zoom in scenes(rep):
        png, wav, mp4 = WORK / f"{name}.png", WORK / f"{name}.wav", WORK / f"{name}.mp4"
        render_png(html, png)
        narrate(text, wav)
        total += clip(png, wav, zoom, mp4)
        clips.append(mp4)
        print(f"  {name}: {total:5.1f} s")

    concat = WORK / "concat.txt"
    concat.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips), encoding="utf-8")
    joined = WORK / "joined.mp4"
    run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(joined)])

    # Synthesised pad (fifth chord, slow beating) mixed under the voice.
    pad = (
        f"aevalsrc='0.05*sin(2*PI*110*t)*(0.6+0.4*sin(2*PI*0.08*t))+0.035*sin(2*PI*164.81*t)"
        f"+0.03*sin(2*PI*220*t)*(0.6+0.4*sin(2*PI*0.05*t+1))+0.02*sin(2*PI*329.63*t)':"
        f"s=48000:d={total:.2f},afade=t=in:d=2,afade=t=out:st={total - 3:.2f}:d=3,"
        f"lowpass=f=900,aformat=channel_layouts=stereo,volume=0.55[pad]"
    )
    final = OUT / "copilote-maintenance-promo.mp4"
    run(
        [
            FFMPEG,
            "-y",
            "-i",
            str(joined),
            "-filter_complex",
            f"{pad};[0:a][pad]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,loudnorm=I=-16:TP=-1.5[a]",
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            str(final),
        ]
    )
    shutil.copy(WORK / "01-titre.png", OUT / "poster.png")
    print(f"Video: {final} ({total:.1f} s, {final.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
