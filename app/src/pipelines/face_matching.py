"""Backward-compatible multi-angle templates and conservative nearest-identity matching."""
import numpy as np

def face_templates(value):
    try:
        array = np.asarray(value, dtype=float)
    except (ValueError, TypeError):
        return []
    if array.shape == (128,):
        array = array.reshape(1, 128)
    if array.ndim != 2 or array.shape[1] != 128:
        return []
    return [row for row in array if np.isfinite(row).all() and np.linalg.norm(row) > 0]

def build_templates(students):
    vectors, identities = [], []
    for student in students:
        for template in face_templates(student.get('face_embedding')):
            vectors.append(template)
            identities.append(int(student['student_id']))
    return vectors, identities

def match_face(encoding, vectors, identities, threshold=0.6, margin=0.04):
    encoding = np.asarray(encoding, dtype=float)
    if encoding.shape != (128,) or not np.isfinite(encoding).all() or not vectors:
        return None, None
    distances = np.linalg.norm(np.asarray(vectors) - encoding, axis=1)
    # Compare identities, not samples: multiple photos of one student aren't competitors.
    by_identity = {}
    for sid, distance in zip(identities, distances):
        by_identity[sid] = min(by_identity.get(sid, float('inf')), float(distance))
    ranked = sorted(by_identity.items(), key=lambda item: item[1])
    sid, distance = ranked[0]
    if distance > threshold or len(ranked) > 1 and ranked[1][1] - distance < margin:
        return None, distance
    return sid, distance

def box_overlap(first, second):
    # Intersection over the smaller box merges different detectors' bounds on one face.
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right-left) * max(0, bottom-top)
    area1 = max(0, first[2]-first[0]) * max(0, first[3]-first[1])
    area2 = max(0, second[2]-second[0]) * max(0, second[3]-second[1])
    return intersection / max(1, min(area1, area2))

def merge_boxes(boxes):
    result = []
    for box in boxes:
        if not any(box_overlap(box, kept) > 0.55 for kept in result):
            result.append(box)
    return result
