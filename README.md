# SnapClass

AI-assisted attendance and classroom management with face recognition and voice identification. Teachers manage courses and review attendance; students register biometric profiles, sign in with FaceID, and enroll in courses.

The project has a Flask landing page and a Streamlit application backed by Supabase.

## Streamlit Cloud deployment

Use branch `main`, entrypoint `app/app.py`, and Python **3.11**. Add
`SUPABASE_URL` and `SUPABASE_KEY` in Streamlit's private Secrets settings.
See [DEPLOYMENT.md](DEPLOYMENT.md#streamlit-community-cloud) for cloud setup,
model availability, database requirements and verification commands.

## Features

- Teacher registration, login, course management, and attendance records.
- Student registration with **three required face photos** (front, left side, right side) and a **required voice sample**.
- Student FaceID login and course enrollment.
- Face attendance from classroom photos, with multi-angle references and lighting correction for usable dark or bright images.
- **Voice-only attendance:** students can say any clear sentence. Spoken roll numbers and daily codes are not required.
- Experimental separation of **two or three overlapping speakers**.
- Separate controls to update face photos and voice samples after registration.
- Review verified students before saving attendance.

## Technology

| Component | Implementation |
| --- | --- |
| Landing page | Flask, HTML, CSS |
| Attendance interface | Streamlit |
| Database | Supabase PostgreSQL |
| Face detection and embeddings | OpenCV, dlib, face_recognition_models |
| Face matching | Nearest identity using 128-dimensional multi-angle embeddings |
| Voice embeddings | Resemblyzer, librosa, PyTorch |
| Overlapping speech | SpeechBrain SepFormer models |
| Voice processing isolation | Python subprocess workers |
| Optional local state | SQLite roll-number registry |

The repository retains an earlier Whisper transcription and daily-challenge implementation. The current attendance screen uses voice identification and skips transcription.

## Local setup on Windows

The existing local installation has been tested with Python 3.10.11. Use a 64-bit Python installation, Git, an internet connection for Supabase and initial model downloads, and a microphone/camera if recording in the browser. Models run on CPU and can take time to load.

### 1. Install dependencies

From the project root in PowerShell:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r app/requirements.txt
.\venv\Scripts\python.exe -m pip install -r landing/requirements.txt
```

Skip virtual-environment creation if the project already has a working `venv`. The face-model dependency is installed from GitHub, so Git must be available during installation. Dependencies include native libraries; installation on other Python versions or operating systems may need adjustments.

### 2. Configure Supabase

1. Create or open your Supabase project.
2. Run [supabase_schema.sql](supabase_schema.sql) in the Supabase SQL editor.
3. Create `app/.streamlit/secrets.toml` with your project settings:

```toml
SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
SUPABASE_KEY = "YOUR_SUPABASE_KEY"
```

The application also reads `SUPABASE_URL` and `SUPABASE_KEY` from environment variables; these take precedence over the secrets file. This project does not require a `.env` file or automatically load one. Keep actual credentials outside Git and never expose a secret/service key in browser code.

The supplied schema creates `teachers`, `students`, `subjects`, `subject_students`, and `attendance_logs`. Face and voice references are stored as JSON embeddings. **The schema disables row-level security and grants broad table access for prototype use.** Review database policies and authentication before exposing a real student database publicly.

### 3. Start both services

```powershell
powershell -ExecutionPolicy Bypass -File .\start_local.ps1
```

| Service | Local address |
| --- | --- |
| Landing page | http://localhost:5002 |
| Attendance application | http://localhost:8501 |

The launcher starts hidden background processes and writes logs and process IDs to `temp/`. It skips ports already in use. Run it from a normal local terminal so the services do not inherit a blocked proxy.

For a foreground session that stops both services with **Ctrl+C**, use:

```powershell
.\venv\Scripts\python.exe run.py
```

Use one launcher at a time. To run only Streamlit:

```powershell
cd app
..\venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

## Using the application

### Student registration

Open **Student Portal**, choose **Register**, enter a name, and capture or upload three different photos. Use a front view and modest left/right turns with the face clearly visible. Record **8?30 seconds** of clear speech, then create the account.

Each image must contain one face, and the side views must match the front photo. Duplicate registration is blocked when at least two views confidently match the same existing student. A single side-view collision does not establish a duplicate. Registered students can sign in to update their full three-photo set or replace their voice sample independently.

### Teacher attendance

Register or sign in through **Teacher Portal**, create a course, and enroll students using the course enrollment workflow. Choose photo attendance or **Use Voice Attendance**, review the results, and save verified attendance.

For individual voice attendance, record **3?20 seconds** of any clear sentence. For overlapping mode, select exactly **2** or **3** speakers and record **3?12 seconds**. Roll numbers are optional labels; the student is identified from the enrolled voice profile.

Voice matching requires cosine similarity of at least **0.75** and a gap of at least **0.08** to the next enrolled voice. These values are thresholds, not accuracy percentages. Unknown, weak, or ambiguous voices are rejected. Duplicate identities recovered from overlapping streams are rejected rather than counted twice. Use the same microphone and a similar distance for enrollment and attendance.

## Recognition limits

- Side views and lighting correction improve the available references but cannot recover a hidden face or severely clipped image.
- Overlapping recognition is experimental and supports two or three speakers, not four or five. Noise and speaker leakage can prevent verification.
- Voice-only attendance does not check a daily phrase and cannot establish liveness against recordings or synthesized voices.
- No classroom accuracy percentage is claimed. Evaluate with representative student recordings and photos before relying on automatic results.
- Native audio processing runs in isolated workers. A worker crash or timeout returns an error without deliberately stopping Streamlit; failed verification adds no attendance.

## Project structure

```text
app/
  app.py                    Streamlit entry point
  requirements.txt          Attendance and model dependencies
  .streamlit/               Theme and local secrets
  src/
    components/             Registration and attendance dialogs
    database/               Supabase configuration and queries
    pipelines/              Face, voice, separation, and worker logic
    screens/                Home, student, and teacher screens
    ui/                     Shared layouts and styling
  tests/                    Automated regression tests
  .local/                   Ignored model cache and SQLite state
landing/                    Flask landing page and assets
supabase_schema.sql         Prototype database schema
start_local.ps1             Background Windows launcher
run.py                      Foreground launcher
render.yaml                 Cloud deployment configuration
DEPLOYMENT.md               Deployment notes and troubleshooting
```

Keep `app/.local/` between restarts to preserve downloaded models and local roll mappings. That local state is not synchronized to Supabase. Secrets, model caches, virtual environments, and runtime logs should remain untracked.

## Tests

Run from the `app` directory with the configured environment:

```powershell
cd app
..\venv\Scripts\python.exe -m unittest discover -s tests
```

The suite covers face matching and lighting, enrollment validation, voice verification, overlapping-stream handling, and worker error recovery. Mocked tests do not establish real-world biometric accuracy.

## Troubleshooting and deployment

- **Browser connection error:** check `temp/attendance.err.log` and restart the local services if Streamlit has stopped.
- **Supabase connection refused:** verify network access and proxy configuration. A traceback involving `http_proxy.py` may indicate an unreachable proxy inherited at launch.
- **Voice rejected:** update the enrollment recording using the same microphone, speak clearly, and retry individual mode.
- **Side-photo registration error:** use smaller turns and clear lighting. If two photos identify an existing account, sign in and update that profile.
- **Slow first analysis:** models may download or load. Overlap separation requires more CPU and memory than individual recognition.

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed local and cloud setup. `render.yaml` provides deployment configuration; its presence does not mean the application has been published or verified in the cloud. Some sections of the deployment notes describe the superseded daily-challenge mode; follow the current voice-only behavior described above.
