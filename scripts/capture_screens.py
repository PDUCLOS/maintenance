"""Capture the real demo screenshots used by the showcase (docs/img/demo-*.png).

Drives the locally installed Microsoft Edge through Playwright (no browser
download: `pip install playwright` then channel="msedge"). The API, the static
demo and the Streamlit UI must already be running:

    uvicorn src.api.main:app --port 18000
    python -m http.server 18080 --directory web
    streamlit run src/ui/streamlit_app.py --server.port 18501   (PYTHONPATH=.)

Every answer shown is produced live by the API at capture time.

Usage: python scripts/capture_screens.py [--only chat,refus,streamlit,swagger]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
API = "http://localhost:18000"
DEMO = f"http://localhost:18080/demo.html?api={API}"
UI = "http://localhost:18501"

QUESTIONS = {
    "chat": "Quelle est la valeur de l'exposant de l'équation de durée de vie pour les roulements à rouleaux ?",
    "refus": "Quel temps fera-t-il demain à Lyon ?",
}


def ask_demo(page: Page, question: str, out: Path) -> None:
    page.goto(DEMO)
    page.wait_for_function(
        "() => document.getElementById('status-api').innerText.includes('online')", timeout=30_000
    )
    page.fill("#input", question)
    page.click("#send")
    # The answer bubble gets a meta line (latency, sources) once the API replies.
    page.wait_for_function(
        "() => /latence API/.test(document.getElementById('chat').innerText)", timeout=180_000
    )
    page.wait_for_timeout(500)
    # The demo re-focuses the input (which scrolls the page): reset every scroller.
    page.evaluate(
        "document.activeElement.blur();"
        "document.querySelectorAll('*').forEach(e => { if (e.scrollTop) e.scrollTop = 0; });"
        "window.scrollTo(0, 0)"
    )
    page.screenshot(path=str(out))
    print(f"  {out.name}")


def streamlit(page: Page, question: str, out: Path) -> None:
    page.goto(UI)
    box = page.get_by_placeholder("Pose ta question", exact=False).first
    box.wait_for(timeout=60_000)
    box.fill(question)
    box.press("Enter")
    # The assistant message ends with a caption "⏱️ 7060 ms · 5 sources".
    page.wait_for_function(
        r"() => /\d+ ms · \d+ sources/.test(document.body.innerText)", timeout=180_000
    )
    page.wait_for_timeout(1500)
    page.locator('[data-testid="stChatMessage"]').last.scroll_into_view_if_needed()
    page.evaluate("window.scrollBy(0, 0)")
    page.screenshot(path=str(out))
    print(f"  {out.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="chat,refus,streamlit,swagger")
    wanted = set(ap.parse_args().only.split(","))
    IMG.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        if "chat" in wanted:
            ask_demo(page, QUESTIONS["chat"], IMG / "demo-chat.png")
        if "refus" in wanted:
            ask_demo(page, QUESTIONS["refus"], IMG / "demo-refus.png")
        if "streamlit" in wanted:
            streamlit(
                page,
                "What is the maximum operating temperature of a TriGard seal?",
                IMG / "demo-streamlit.png",
            )
        if "swagger" in wanted:
            page.goto(f"{API}/docs")
            page.wait_for_selector(".opblock", timeout=30_000)
            page.screenshot(path=str(IMG / "demo-swagger.png"))
            print("  demo-swagger.png")
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
