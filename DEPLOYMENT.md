# Deploy SnapClass

## Streamlit Community Cloud

Deploy the attendance app from this repository with:

- Repository: `Yash-Bharambe/Voice-n-Face-rogonition-based-attendence-system-`
- Branch: `main`
- Main file: `app/app.py`
- Python: **3.11** in Advanced settings.
- Secrets: copy your own `SUPABASE_URL` and `SUPABASE_KEY` into the private
  Secrets field. Never commit the actual secrets file.

The root `.streamlit/config.toml` configures Community Cloud; dependencies are
in `app/requirements.txt`, and `packages.txt` supplies libsndfile. Linux installs
CPU-only PyTorch. The small YuNet model and its license are bundled in
`app/models/yunet`; Resemblyzer ships its encoder, and experimental SepFormer
models download on first use. Overlap inference can exceed free-tier memory or
take several minutes. Start with individual voice attendance.

Community Cloud hosts the Streamlit attendance app. The separate Flask landing
page requires a separate host, such as the existing Render blueprint below.
Local SQLite roll labels and downloaded model caches are ephemeral in the cloud;
attendance, courses and biometric profiles persist in Supabase.

Before publishing, use a dedicated demonstration database. The supplied schema
has broad prototype permissions; enforce database access policies before using
real student records. The GitHub workflow runs regression and application startup
checks on Ubuntu. To reproduce locally from the repository root:

```powershell
.\venv\Scripts\python.exe scripts/verify_cloud.py
cd app
..\venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Current voice attendance behavior

Attendance now identifies enrolled students by their voice alone. Students may
say any clear sentence for 3–20 seconds individually, or 3–12 seconds in groups
of two or three. Spoken roll numbers and daily challenge codes are no longer
required. Roll numbers are optional display labels. Transcription is skipped;
voice similarity must still reach 0.75 with a 0.08 gap to competing profiles.
Weak, ambiguous, and duplicate separated voices are rejected. Processing stays
in isolated workers. This mode does not provide the old daily-code check against
previously recorded speech. The challenge/transcription descriptions below
describe the earlier mode and are superseded by this section.

The repository contains two Python web services. `render.yaml` configures both
for Render. The AI service's memory requirements depend on uploaded images and
audio; upgrade its instance if the hosting logs show memory exhaustion.

1. Create a Supabase project and run `supabase_schema.sql` in its SQL editor.
2. Push these deployment changes to your GitHub repository.
3. In Render, select **New > Blueprint** and connect this repository.
4. Set `SUPABASE_URL` and `SUPABASE_KEY` for the attendance service using your
   Supabase project URL and anon key. Keep credentials outside Git.
5. Set the landing service's `ATTENDANCE_APP_URL` to the attendance service's
   actual HTTPS URL from Render. Redeploy the landing service after setting it.
6. Verify the landing page's three attendance buttons, student enrollment,
   teacher login, and photo/voice attendance against your database.

For local use, install each service's requirements and use `python run.py`.
The landing page defaults to `http://localhost:8501`. The attendance service
accepts environment variables or `app/.streamlit/secrets.toml`.

Configuration prepared locally is not a completed cloud deployment. Render
account access and Supabase configuration are required to publish and verify it.

## Local deployment

Use `powershell -ExecutionPolicy Bypass -File .\start_local.ps1` to start both
services in the background. Open http://localhost:5002 for the landing page or
http://localhost:8501 for attendance. Logs and process IDs are in `temp/`.
Supabase settings are stored in the ignored `app/.streamlit/secrets.toml`.

## Voice attendance with daily challenge

Students sign in with FaceID, open **Update voice sample** to replace their voice, and use
**Voice attendance roll number** to set a numeric roll number. Teachers can also set roll numbers under
**Use Voice Attendance > Set student roll numbers**. These mappings and challenge
codes persist locally in the ignored `app/.local/voice_state.sqlite3`.

Each course gets a cryptographically random four-word code each day, using
India time. Teachers share it only during attendance and can replace it to
invalidate an earlier challenge. Students say exactly:
**Roll number two three, code [the four displayed words]**.
If transcription omits the separator word "code", the phrase is still accepted
when the roll number and challenge words pass the same checks.
One incorrect or missing challenge word is allowed: at least three of the four
words must match in order. Extra words, rearrangements, and two errors are
rejected. Roll numbers are exact, and voice similarity (0.75 minimum) and the
gap to competing voices (0.08 minimum) remain unchanged. This tolerance applies
to both individual and overlapping recordings. Replacement codes differ from
the previous code in at least two positions so an old code cannot pass through
the one-word tolerance. This is word tolerance, not a measured accuracy rate.
Use English and speak roll digits separately. Record one student at a time for
3?20 seconds. The app transcribes locally with faster-whisper small.en, checks
the complete phrase, verifies the claimed student's enrolled voice, and rejects
uncertain matches. Review results, then save only verified students as present.
Students who fail verification remain pending rather than being marked absent.

Daily rollover, replacing the code, changing the course, or changing roll
assignments clears pending verifications. A repeated identical successful clip
is rejected within the current browser session. The code is a spoken challenge,
not a password: it rejects older wrong-code recordings but does not establish
liveness against voice cloning, spliced audio, or same-day replay in another
session. Experimental two- and three-speaker overlap mode is available; four and five speakers remain unsupported. The initial voice
similarity threshold (0.75) and competitor margin (0.08) need calibration with
real student recordings; they are not measured accuracy guarantees.

The transcription model downloads once into `app/.local/models` and runs on CPU.
Keep `app/.local` when restarting; it is local state, not synchronized to Supabase.

## Overlapping voice attendance (experimental)

In **Use Voice Attendance**, choose **Overlapping group (experimental)** and
select exactly **2** or **3** students. Each student says their own complete
roll-number phrase and the displayed daily code. Record a 3?12 second group
clip. Processing separates the mixture with the matching SpeechBrain SepFormer
WSJ0-2Mix or WSJ0-3Mix model at 8 kHz, resamples each source to 16 kHz, and applies
the same transcription, code, roll, and voice verification checks independently.
The source separation models run on CPU and may take significant time and RAM.

The stream report shows what was heard and which voices require a retry.
Only students who pass all checks enter the verified list. A student appearing
in multiple separated streams is rejected in all of those streams to avoid
counting source leakage as independent evidence. Retry uncertain students using
one-student mode, then review and save verified attendance.

The selected count is supplied by the teacher; the model does not automatically
count speakers. Four or five simultaneous voices are not supported. No accuracy
claim is made for classroom acoustics, overlapping identical challenge words,
or very short speech. Real recordings with known participants are needed to
measure false acceptance and missed students before operational use. Existing
replay and voice-cloning limitations still apply.

## Side-facing face attendance

Face scanning now merges YuNet detections, the original dlib frontal detector,
and left/right profile-cascade detections. Mirrored profile boxes are mapped
back to the original image and duplicate boxes are merged. The original dlib
128-dimensional identity embeddings are retained, so old student profiles work.

For better angled recognition, students sign in and open **Update face images**. Capture or upload a complete
front, left-side, and right-side set (one face per image), then save it. Each new reference must match that student's existing identity before
being added; start with smaller turns if an extreme profile cannot be confirmed.
Up to seven references are stored in the existing Supabase JSONB face_embedding
field, with backward-compatible support for old single-vector records.

Recognition chooses the closest reference across each identity, requires distance
at most 0.6, and rejects competing identities within a 0.04 distance margin.
Detecting a profile alone never establishes attendance. Complete profiles,
blur, occlusion, small faces, and unsuitable reference photos can still prevent
recognition. Side-view classroom accuracy has not been measured.

The YuNet ONNX file and its MIT license are bundled in `app/models/yunet`.
The profile cascade is bundled with OpenCV.

## Bright and dark classroom photos

Face analysis applies bounded gamma correction and mild CLAHE on the luminance
channel when an image or a detected face is dark, bright, or low contrast.
Detection runs on the original and corrected image when correction is needed;
duplicate boxes are merged. Recognition compares original and corrected views
of each face against the same enrollment templates at the unchanged 0.6 distance
threshold and 0.04 competitor margin. Conflicting identities between views are
rejected. Enrollment uses the corrected reference when needed.

Photos with virtually no visible detail are skipped. Faces with mostly clipped
black/white pixels or too little contrast are rejected. The teacher sees lighting
and clipping notices; if no students can be confidently recognized, the app does
not generate an all-absent attendance result. When some students are recognized,
the existing attendance review still applies: review unmatched students carefully
and retake unsuitable photos before saving absences.

These adjustments cannot recover details erased by complete darkness, saturated
highlights, glare, or motion blur. Validation includes synthetic exposure changes
to an example photograph, not measured accuracy for real classroom lighting.

## Required registration and profile replacement

The student portal now has explicit **Login** and **Register** choices. A new
student must provide a name, three different front/left/right photos, and an
8?30 second voice recording containing at least four seconds of usable speech.
Camera capture and image upload are supported. Each photo must contain exactly
one recognizable face; the side views must match the front reference. Existing
registration is blocked as a duplicate when at least two distinct views
confidently match the same registered student. A single side-view collision
does not block registration. The existing face-distance and competing-identity
margin checks apply; all three photos must still match the front view. All inputs
are validated before the single student insert, and the database helper rejects
missing face or voice templates as well.

Signed-in students have separate **Update face images** and **Update voice
sample** controls. Face replacement requires a complete new three-photo set
matching the logged-in student's existing identity. Voice replacement requires a
new valid recording. Current templates remain stored unless validation succeeds
and the database returns an updated row. Old accounts continue to work and can
use these controls to complete or replace their biometric profiles. Roll-number
editing is separate under **Voice attendance roll number**.

Photo labels guide the required views; the app checks distinct files and face
identity, rather than certifying an exact head angle. Biometric references are
stored as embeddings, not an image/audio gallery.

## Voice recording quality and diagnostics

Transcription now uses the larger `small.en` local Whisper model. Enrollment and
attendance remove DC offset and adjust audio volume with at most 20 dB of gain,
while retaining speech duration, silence, clipping, transcription-confidence,
voice-identity, and daily-code checks. Recordings with sustained clipping are
rejected; increasing volume cannot repair noise or recover clipped speech.

Voice mismatch errors now show the actual cosine similarity, the required 0.75
threshold, or the gap to competing voices (required 0.08). These scores are not
accuracy percentages. Use the same microphone and distance for enrollment and
attendance, update poor enrollment samples, and first test one-student mode.
The larger transcription model uses more RAM and CPU time. No measured classroom
accuracy improvement is claimed without labelled student recordings.

## Audio analysis disconnects

On this Windows installation, voice analysis caused a native Python access
violation in `MSVCP140.dll`, terminating Streamlit without a Python traceback.
Voice transcription, overlap separation, and enrollment now run in isolated
subprocesses. Recordings and profile data travel through pipes without temporary
audio files. Worker crashes and ten-minute timeouts return an error in the app;
failed analysis adds no attendance. Worker exit codes and diagnostics are logged
in `temp/attendance.err.log`. Each recording starts a fresh worker and reloads
its models, so analysis can take longer, especially for overlapping voices.

If the browser still displays the old connection error, reload
`http://localhost:8501`. To start the local services again, run `./start_local.ps1`
from the project directory.

If Supabase requests fail with `WinError 10061` and the traceback includes
`http_proxy.py`, check the server's launch environment. A restricted tool session
can inject an unreachable `127.0.0.1:9` proxy. Restart the services from your normal
local PowerShell terminal or an approved unrestricted environment. The launcher
now rejects that blocked proxy before starting services; legitimate proxy
configuration is preserved.
