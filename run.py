import os
import sys
import time
import subprocess
import signal

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(ROOT_DIR, "venv", "Scripts", "python.exe")
PYTHON_EXE = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable

def start_services():
    print("=" * 60)
    print("      🚀 STARTING SNAPCLASS AI ATTENDANCE SYSTEM 🚀      ")
    print("=" * 60)
    print(f"Using Python: {PYTHON_EXE}")
    print(f"Project Directory: {ROOT_DIR}\n")

    # 1. Start Landing Page (Flask - Port 5002)
    print("🌐 Starting Landing Page (Flask) on http://localhost:5002 ...")
    landing_script = os.path.join(ROOT_DIR, "landing", "app.py")
    landing_proc = subprocess.Popen(
        [PYTHON_EXE, landing_script],
        cwd=os.path.join(ROOT_DIR, "landing")
    )

    # Give landing page a brief moment to start
    time.sleep(1.5)

    # 2. Start AI Attendance App (Streamlit - Port 8501)
    print("🤖 Starting AI Attendance App (Streamlit) on http://localhost:8501 ...")
    app_script = os.path.join(ROOT_DIR, "app", "app.py")
    streamlit_proc = subprocess.Popen(
        [PYTHON_EXE, "-m", "streamlit", "run", app_script, "--server.port", "8501"],
        cwd=os.path.join(ROOT_DIR, "app")
    )

    print("\n" + "=" * 60)
    print("  ✅ BOTH SERVICES ARE RUNNING!")
    print("  👉 Landing Page:         http://localhost:5002")
    print("  👉 AI Attendance App:    http://localhost:8501")
    print("  Press CTRL+C anytime to stop both services.")
    print("=" * 60 + "\n")

    def handle_shutdown(sig=None, frame=None):
        print("\nStopping services...")
        try:
            landing_proc.terminate()
        except Exception:
            pass
        try:
            streamlit_proc.terminate()
        except Exception:
            pass
        print("All services stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        while True:
            time.sleep(1)
            # Check if either process crashed
            if landing_proc.poll() is not None:
                print("Landing page process exited.")
                break
            if streamlit_proc.poll() is not None:
                print("Streamlit app process exited.")
                break
    except KeyboardInterrupt:
        handle_shutdown()

if __name__ == "__main__":
    start_services()
