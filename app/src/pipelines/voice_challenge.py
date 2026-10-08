"""Local roll registry and per-course daily random challenges."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import secrets
import sqlite3
from contextlib import contextmanager

STATE_PATH = Path(__file__).resolve().parents[2] / '.local' / 'voice_state.sqlite3'
INDIA = timezone(timedelta(hours=5, minutes=30))
WORDS = 'apple river silver garden yellow window pencil orange planet purple tiger lemon summer winter music coffee paper forest rocket basket flower ocean monkey rabbit cherry table candle button cotton pocket sunset morning'.split()

def today():
    return datetime.now(INDIA).date().isoformat()

@contextmanager
def connect():
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(STATE_PATH, timeout=10)
    db.execute('CREATE TABLE IF NOT EXISTS rolls (student_id INTEGER PRIMARY KEY, roll TEXT UNIQUE NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS challenges (subject_id INTEGER, day TEXT, code TEXT, PRIMARY KEY(subject_id, day))')
    try:
        with db:
            yield db
    finally:
        db.close()

def normalize_roll(value):
    value = str(value).strip()
    if not re.fullmatch(r'\d{1,12}', value):
        raise ValueError('Roll numbers must contain 1–12 digits.')
    return str(int(value))

def get_rolls():
    with connect() as db:
        return dict(db.execute('SELECT student_id, roll FROM rolls'))

def save_rolls(mapping):
    normalized = {int(sid): normalize_roll(roll) for sid, roll in mapping.items() if str(roll).strip()}
    if len(set(normalized.values())) != len(normalized):
        raise ValueError('Each student must have a different roll number.')
    with connect() as db:
        db.executemany('DELETE FROM rolls WHERE student_id=?', [(int(sid),) for sid in mapping])
        try:
            db.executemany('INSERT INTO rolls VALUES (?, ?)', normalized.items())
        except sqlite3.IntegrityError as exc:
            raise ValueError('A roll number is already assigned to another student.') from exc

def get_challenge(subject_id, rotate=False):
    day = today()
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT code FROM challenges WHERE subject_id=? AND day=?', (subject_id, day)).fetchone()
        if row and not rotate:
            return day, row[0]
        while True:
            code = ' '.join(secrets.SystemRandom().sample(WORDS, 4))
            if not row or sum(a == b for a, b in zip(code.split(), row[0].split())) < 3:
                break
        db.execute('INSERT OR REPLACE INTO challenges VALUES (?, ?, ?)', (subject_id, day, code))
    return day, code

DIGITS = dict(zip('zero one two three four five six seven eight nine'.split(), range(10)))
SMALL = dict(zip('ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(), range(10, 20)))
TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10)))

def spoken_number(text):
    tokens = text.strip().split()
    if tokens and all(t.isdigit() or t in DIGITS for t in tokens):
        return normalize_roll(''.join(t if t.isdigit() else str(DIGITS[t]) for t in tokens))
    if len(tokens) == 1 and tokens[0] in SMALL:
        return str(SMALL[tokens[0]])
    if tokens and tokens[0] in TENS and (len(tokens) == 1 or len(tokens) == 2 and tokens[1] in DIGITS):
        return str(TENS[tokens[0]] + (DIGITS[tokens[1]] if len(tokens) == 2 else 0))
    raise ValueError('Say the roll number digit by digit.')

def parse_claim(transcript, expected_code):
    text = ' '.join(re.sub(r'[^a-z0-9\s]', ' ', transcript.lower()).split())
    # "Role" is acoustically identical to "roll" and commonly emitted by ASR.
    number_token = r'(?:[0-9]+|' + '|'.join([*DIGITS, *SMALL, *TENS]) + r')'
    match = re.fullmatch(r'(?:my )?(?:roll|role) (?:number|no) (' + number_token +
                        r'(?: ' + number_token + r')*) (?:(?:code|cold) )?([a-z]+(?: [a-z]+){2,3})', text)
    if not match:
        raise ValueError('Say exactly: Roll number [digits], code [four challenge words].')
    spoken = match[2].split()
    expected = expected_code.split()
    # Permit one substitution or omission, retaining order. Extra words and
    # rearrangements are rejected; roll numbers and voice gates remain exact.
    valid = len(expected) == 4 and (
        len(spoken) == 4 and sum(a == b for a, b in zip(spoken, expected)) >= 3 or
        len(spoken) == 3 and any(spoken == expected[:i] + expected[i+1:] for i in range(4)))
    if not valid:
        raise ValueError('At least three challenge words must match in order, with at most one incorrect or missing word. Record again.')
    return spoken_number(match[1])

def verify_voice(embedding, candidates, threshold=0.75, margin=0.08):
    import numpy as np
    vector = np.asarray(embedding, dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
        raise ValueError('No usable voice signature. Record again.')
    matches = []
    for key, student in candidates.items():
        stored = np.asarray(student.get('voice_embedding'), dtype=float)
        if stored.shape != vector.shape or not np.isfinite(stored).all() or np.linalg.norm(stored) == 0:
            continue
        score = float(np.dot(vector, stored) / (np.linalg.norm(vector) * np.linalg.norm(stored)))
        matches.append((score, key, student['student_id']))
    matches.sort(key=lambda item: item[0], reverse=True)
    if not matches:
        raise ValueError('No valid enrolled voice profiles in this course.')
    score, key, sid = matches[0]
    if score < threshold:
        raise ValueError(f'Voice similarity {score:.3f} is below the required {threshold:.2f}. Update the voice sample using this microphone and retry.')
    if len(matches) > 1 and score - matches[1][0] < margin:
        raise ValueError('This voice is too close to another enrolled voice. Record again individually.')
    return sid, score, key

def verify_claim(transcript, expected_code, embedding, candidates, threshold=0.75, margin=0.08):
    import numpy as np
    roll = parse_claim(transcript, expected_code)
    claimed = candidates.get(roll)
    if not claimed or not claimed.get('voice_embedding'):
        raise ValueError('This roll number has no enrolled voice profile in this course.')
    vector = np.asarray(embedding, dtype=float)
    def similarity(stored):
        other = np.asarray(stored, dtype=float)
        if other.shape != vector.shape or not np.isfinite(other).all() or not np.isfinite(vector).all():
            raise ValueError('Invalid voice profile. Enroll the voice again.')
        denominator = np.linalg.norm(vector) * np.linalg.norm(other)
        return float(np.dot(vector, other) / denominator) if denominator else 0.0
    score = similarity(claimed['voice_embedding'])
    competitors = [similarity(s['voice_embedding']) for r, s in candidates.items() if r != roll and s.get('voice_embedding')]
    if score < threshold:
        raise ValueError(f'Voice similarity {score:.3f} is below the required {threshold:.2f}. Update the voice sample using this microphone, then retry in one-student mode.')
    if competitors and score - max(competitors) < margin:
        raise ValueError(f'Voice similarity {score:.3f} is too close to another enrolled voice (gap {score-max(competitors):.3f}; required {margin:.2f}). Record again individually.')
    return claimed['student_id'], score, roll
