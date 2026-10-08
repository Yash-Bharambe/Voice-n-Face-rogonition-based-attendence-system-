import subprocess
import time
import urllib.request
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = os.path.join(ROOT_DIR, "venv", "Scripts", "python.exe")

def test_landing():
    print("Testing Flask Landing Page...")
    landing_script = os.path.join(ROOT_DIR, "landing", "app.py")
    proc = subprocess.Popen(
        [PYTHON_EXE, landing_script],
        cwd=os.path.join(ROOT_DIR, "landing")
    )
    time.sleep(3)
    try:
        req = urllib.request.urlopen("http://localhost:5002", timeout=5)
        status = req.getcode()
        content = req.read().decode('utf-8')
        print(f"Landing page response status: {status}")
        assert "SnapClass" in content, "SnapClass title not found in HTML"
        print(" Landing page test passed!")
    finally:
        proc.terminate()
        proc.wait()

def test_streamlit():
    print("\nTesting Streamlit App...")
    app_script = os.path.join(ROOT_DIR, "app", "app.py")
    proc = subprocess.Popen(
        [PYTHON_EXE, "-m", "streamlit", "run", app_script, "--server.port", "8501", "--server.headless", "true"],
        cwd=os.path.join(ROOT_DIR, "app")
    )
    time.sleep(10)
    try:
        req = urllib.request.urlopen("http://localhost:8501", timeout=5)
        status = req.getcode()
        print(f"Streamlit health response status: {status}")
        assert status == 200, "Streamlit health check did not return 200"
        print(" Streamlit app test passed!")
    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    test_landing()
    test_streamlit()
    print("\n ALL LOCAL SERVER TESTS PASSED SUCCESSFULLY!")
