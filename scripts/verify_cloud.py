"""Check the real app from the repository-root working directory, without DB writes."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "app"))
# Syntactically valid placeholders: these smoke checks must not contact a real DB.
os.environ["SUPABASE_URL"] = "https://deployment-smoke-test.supabase.co"
os.environ["SUPABASE_KEY"] = "deployment-smoke-test-placeholder"

from streamlit.testing.v1 import AppTest


def check_app():
    app = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=90).run()
    assert not app.exception, [e.message for e in app.exception]
    assert {b.label for b in app.button} >= {"Student Portal", "Teacher Portal"}
    for portal in ("Student Portal", "Teacher Portal"):
        screen = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=90).run()
        next(b for b in screen.button if b.label == portal).click().run()
        assert not screen.exception, [e.message for e in screen.exception]
    from src.pipelines.face_pipeline import load_pose_detectors, load_dlib_models
    detector, profile = load_pose_detectors()
    assert detector is not None and not profile.empty()
    assert len(load_dlib_models()) == 3
    print("Cloud smoke checks passed: root entrypoint, both portals, YuNet and dlib models.")


if __name__ == "__main__":
    check_app()
