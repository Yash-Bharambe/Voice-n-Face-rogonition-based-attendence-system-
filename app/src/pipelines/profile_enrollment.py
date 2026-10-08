import hashlib
import io
from collections import Counter
import numpy as np
from PIL import Image
from src.pipelines.face_pipeline import get_face_embeddings
from src.pipelines.face_matching import build_templates, match_face
from src.pipelines.voice_service import enroll_voice_sample

ANGLES = ('Front', 'Left side', 'Right side')

def prepare_face_profile(photos, students, student_id=None):
    if len(photos) != 3 or any(not photo for photo in photos):
        raise ValueError('Provide all three photos: front, left side, and right side.')
    if len({hashlib.sha256(photo).digest() for photo in photos}) != 3:
        raise ValueError('Use three different photos: front, left side, and right side.')
    references = []
    matched_identities = []
    vectors, identities = build_templates(students)
    for angle, photo in zip(ANGLES, photos):
        try:
            image = np.array(Image.open(io.BytesIO(photo)).convert('RGB'))
        except Exception as exc:
            raise ValueError(f'{angle}: could not read the photo.') from exc
        embeddings = get_face_embeddings(image)
        if len(embeddings) != 1:
            raise ValueError(f'{angle}: use a clear photo containing exactly one face.')
        vector = embeddings[0]
        sid, _ = match_face(vector, vectors, identities)
        if student_id is not None and sid != student_id:
            raise ValueError(f'{angle}: could not confirm this is your face. Use a clearer photo or a smaller turn.')
        if references and np.linalg.norm(vector-references[0]) > 0.6:
            raise ValueError(f'{angle}: does not match the front photo confidently. Retake with a smaller turn.')
        references.append(vector)
        matched_identities.append(sid)
    if student_id is None:
        # A single side-view collision is insufficient evidence of a duplicate.
        # Use the same distance and competing-identity margin as face matching,
        # requiring agreement across at least two distinct enrollment views.
        matches = Counter(sid for sid in matched_identities if sid is not None)
        if any(count >= 2 for count in matches.values()):
            raise ValueError('At least two photos match the same registered student. Sign in to update that profile. If this is a new student, retake clear photos with smaller side turns.')
    return [vector.tolist() for vector in references]

def prepare_registration(name, photos, voice_bytes, students):
    if not name.strip():
        raise ValueError('Enter your name.')
    if len(photos) != 3 or any(not photo for photo in photos):
        raise ValueError('Front, left-side, and right-side photos are all required.')
    if not voice_bytes:
        raise ValueError('A voice recording is required to register.')
    references = prepare_face_profile(photos, students)
    voice = enroll_voice_sample(voice_bytes)
    return name.strip(), references, voice
