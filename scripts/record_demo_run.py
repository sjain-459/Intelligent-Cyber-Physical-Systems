"""
Runs the real CTMAS pipeline once and records its full WebSocket event
stream to a static JSON file, plus the training-metrics plot it produces.

This backs the "demo replay" mode: the deployed dashboard has no live
backend (avoids needing a server with enough RAM to hold torch/opacus/shap
resident just to serve a public demo link), and instead replays this
faithful recording of one real run with the same pacing the live WebSocket
would have used. Local development (`python api.py` + `npm run dev`)
still runs the pipeline live and is unaffected by this.

Usage:
    python scripts/record_demo_run.py
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import run_simulation_stream  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "public")


def main():
    print("Running the pipeline once to record a demo event stream...")
    events = list(run_simulation_stream())

    os.makedirs(OUT_DIR, exist_ok=True)
    events_path = os.path.join(OUT_DIR, "demo_run.json")
    with open(events_path, "w") as f:
        json.dump(events, f)
    print(f"Wrote {len(events)} events to '{events_path}'")

    metrics_plot = os.path.join("results", "federated_metrics.png")
    if os.path.exists(metrics_plot):
        dest = os.path.join(OUT_DIR, "federated_metrics.png")
        shutil.copyfile(metrics_plot, dest)
        print(f"Copied '{metrics_plot}' -> '{dest}'")


if __name__ == "__main__":
    main()
