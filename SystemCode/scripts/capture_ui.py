"""Screenshot every web app use case (Playwright, the installed Edge; no browser download).

Start the app first, then:
    uv run uvicorn backend.app:app
    uv run --with playwright python scripts/capture_ui.py [--url http://127.0.0.1:8000] [--jev]

Writes reports/figures/ui/*.png. The compare page uses LAYA (free); --jev adds one paid JEV call.
Tooltips and the expandable request boxes are left closed.
"""
import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
CLIPS = {  # 2019 LA eval clips D2 scores far on the right side, and a modern deepfake
    "bonafide": ROOT / "data/raw/ASVspoof2019/LA/ASVspoof2019_LA_eval/flac/LA_E_9082189.flac",
    "spoof": ROOT / "data/raw/ASVspoof2019/LA/ASVspoof2019_LA_eval/flac/LA_E_4640620.flac",
    "own_deepfake": ROOT / "data/raw/bv/boldvoice_brit.wav",
}
OUT = ROOT / "reports/figures/ui"


def shot(page: Page, name: str) -> None:
    page.mouse.move(0, 0)  # no hover tooltip
    page.screenshot(path=OUT / f"{name}.png", full_page=True)
    print(f"saved {name}.png")


def detect(page: Page, clip: Path, system: str = "d2") -> None:
    page.set_input_files("#file", str(clip))
    page.select_option("#single-system", system)
    page.click("#submit")
    page.wait_for_selector("#result:not([hidden]), #error:not([hidden])", timeout=180_000)


def compare(page: Page, clip: Path, judge: str) -> None:
    page.set_input_files("#compare-file", str(clip))
    page.select_option("#compare-model-a", "a2")
    page.select_option("#compare-model-b", "d2")
    page.select_option("#compare-judge", judge)
    page.click("#compare-submit")
    page.wait_for_selector("#compare-result:not([hidden]), #compare-error:not([hidden])",
                           timeout=300_000)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--jev", action="store_true", help="also capture a JEV arbitration (paid)")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge")
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(a.url)
        page.wait_for_selector("#single-system option", state="attached")

        shot(page, "01_detect_upload")
        page.click("[data-mode=record]")
        shot(page, "02_detect_record")
        page.click("[data-mode=upload]")
        for i, (name, clip) in enumerate(CLIPS.items(), start=3):
            detect(page, clip)
            shot(page, f"0{i}_detect_{name}")

        page.click("[data-view=compare]")
        page.wait_for_selector("#compare-judge option", state="attached")
        shot(page, "06_compare_empty")
        compare(page, CLIPS["own_deepfake"], "laya")
        shot(page, "07_compare_laya")
        if a.jev:
            compare(page, CLIPS["own_deepfake"], "jev")
            shot(page, "08_compare_jev")

        page.click("[data-view=rankings]")
        page.wait_for_selector("#rankings-tabs .filter-btn")
        for btn in page.query_selector_all("#rankings-tabs .filter-btn"):
            btn.click()
            label = "".join(c if c.isalnum() else "_" for c in btn.inner_text().strip().lower())
            shot(page, f"09_rankings_{label}")

        page.click("#theme-toggle")
        page.click("[data-view=single]")
        shot(page, "10_detect_dark")
        browser.close()


if __name__ == "__main__":
    main()
