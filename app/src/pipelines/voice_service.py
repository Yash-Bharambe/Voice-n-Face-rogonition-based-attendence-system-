"""Keep native speech libraries outside the Streamlit server process."""
import base64
import json
import logging
from pathlib import Path
import subprocess
import sys


WORKER = Path(__file__).with_name('voice_worker.py')
TIMEOUT_SECONDS = 600


def _run(operation, audio_bytes, **options):
    request = dict(operation=operation, audio=base64.b64encode(audio_bytes).decode('ascii'), **options)
    # Pipes keep recordings and biometric profiles off disk. Each request has an
    # independent process: a native DLL crash cannot kill the web server.
    process = subprocess.Popen(
        [sys.executable, str(WORKER)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(WORKER.parents[2]),
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    try:
        output, errors = process.communicate(json.dumps(request).encode('utf-8'), timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise ValueError('Voice analysis took too long. Try a shorter recording or one student at a time.') from None
    if process.returncode:
        logging.getLogger(__name__).error('Voice worker exited with code %s. Diagnostics: %s',
                                          process.returncode, errors.decode('utf-8', errors='replace')[-4000:])
        raise ValueError('The audio engine stopped unexpectedly. The app is still running; record again or try one student at a time. No attendance was added.')
    try:
        response = json.loads(output)
    except (ValueError, UnicodeError):
        raise ValueError('The audio engine returned an invalid response. Record again.') from None
    if 'error' in response:
        raise ValueError(response['error'])
    return response['result']


def transcribe_claim(audio_bytes):
    transcript, embedding = _run('transcribe', audio_bytes)
    return transcript, embedding


def enroll_voice_sample(audio_bytes):
    return _run('enroll', audio_bytes)

def identify_voices(audio_bytes, candidates, speaker_count=1):
    profiles = {str(key): {field: student.get(field) for field in ('student_id', 'voice_embedding')}
                for key, student in candidates.items()}
    return _run('identify' if speaker_count == 1 else 'identify_overlap', audio_bytes,
                candidates=profiles, speaker_count=speaker_count)


def verify_overlapping_audio(audio_bytes, speaker_count, code, candidates):
    profiles = {roll: {key: student.get(key) for key in ('student_id', 'voice_embedding')}
                for roll, student in candidates.items()}
    return _run('overlap', audio_bytes, speaker_count=speaker_count, code=code, candidates=profiles)
