import dlib
import numpy as np
import cv2
import face_recognition_models
import streamlit as st
from pathlib import Path
from threading import Lock
from src.database.db import get_all_students
from src.pipelines.face_matching import build_templates, match_face, merge_boxes
from src.pipelines.face_lighting import normalize_lighting, usable_face_crop

APP_DIRECTORY = Path(__file__).resolve().parents[2]
MODEL_PATH = APP_DIRECTORY / 'models/yunet/face_detection_yunet_2023mar.onnx'
DETECTOR_LOCK = Lock()

@st.cache_resource
def load_dlib_models():
    return (dlib.get_frontal_face_detector(),
            dlib.shape_predictor(face_recognition_models.pose_predictor_model_location()),
            dlib.face_recognition_model_v1(face_recognition_models.face_recognition_model_location()))

@st.cache_resource
def load_pose_detectors():
    if not MODEL_PATH.is_file():
        raise RuntimeError('YuNet model is missing from app/models/yunet. Restore the bundled model.')
    yunet = cv2.FaceDetectorYN.create(str(MODEL_PATH), '', (320, 320), 0.85, 0.3, 5000)
    profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    if profile.empty():
        raise RuntimeError('The profile face detector is missing from OpenCV.')
    return yunet, profile

def detect_face_boxes(image_np):
    image = np.asarray(image_np)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Provide an RGB image with 8-bit pixels.')
    image = np.ascontiguousarray(image)
    height, width = image.shape[:2]
    if min(height, width) < 30:
        return []
    # Bound detection cost for large classroom photos, then encode on original pixels.
    scale = min(1.0, 1600 / max(height, width))
    small = cv2.resize(image, (round(width*scale), round(height*scale))) if scale < 1 else image
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    yunet, profile = load_pose_detectors()
    boxes = []
    with DETECTOR_LOCK:
        yunet.setInputSize((small.shape[1], small.shape[0]))
        _, faces = yunet.detect(bgr)
        if faces is not None:
            for face in sorted(faces, key=lambda row: -row[-1]):
                x, y, w, h = face[:4]
                boxes.append((x, y, x+w, y+h))
        # The profile cascade searches both directions via a mirrored pass.
        for mirrored in (False, True):
            source = cv2.flip(gray, 1) if mirrored else gray
            for x, y, w, h in profile.detectMultiScale(source, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
                if mirrored:
                    x = small.shape[1] - x - w
                boxes.append((x, y, x+w, y+h))
    frontal, _, _ = load_dlib_models()
    for rect in frontal(small, 1):
        boxes.append((rect.left(), rect.top(), rect.right()+1, rect.bottom()+1))
    bounded = []
    for left, top, right, bottom in boxes:
        box = (max(0, int(left/scale)), max(0, int(top/scale)),
               min(width, int(np.ceil(right/scale))), min(height, int(np.ceil(bottom/scale))))
        if box[2]-box[0] >= 20 and box[3]-box[1] >= 20:
            bounded.append(box)
    return merge_boxes(bounded)

def get_face_embedding_groups(image_np):
    image = np.ascontiguousarray(image_np)
    corrected, info = normalize_lighting(image)
    if not info['usable']:
        return []
    _, predictor, encoder = load_dlib_models()
    boxes = detect_face_boxes(image)
    changed = not np.array_equal(image, corrected)
    if changed:
        boxes = merge_boxes(boxes + detect_face_boxes(corrected))
    groups = []
    for left, top, right, bottom in boxes:
        # Keep a detected but unreadable face in the count; never use an invented embedding.
        if not usable_face_crop(image[top:bottom, left:right]):
            groups.append([])
            continue
        rect = dlib.rectangle(left, top, right-1, bottom-1)
        variants = [corrected, image] if changed else [image]
        face_corrected, _ = normalize_lighting(image[top:bottom, left:right])
        if not np.array_equal(face_corrected, image[top:bottom, left:right]):
            local_variant = image.copy()
            local_variant[top:bottom, left:right] = face_corrected
            variants.insert(0, local_variant)
        group = []
        for variant in variants:
            shape = predictor(variant, rect)
            descriptor = np.asarray(encoder.compute_face_descriptor(variant, shape, 3))
            if descriptor.shape == (128,) and np.isfinite(descriptor).all():
                group.append(descriptor)
        groups.append(group)
    return groups

def get_face_embeddings(image_np):
    # Enrollment stores one corrected reference per physical face.
    groups = get_face_embedding_groups(image_np)
    if any(not group for group in groups):
        raise ValueError('A face is too dark or washed out. Retake the photo with more even lighting.')
    return [group[0] for group in groups]

@st.cache_data(ttl=60)
def get_trained_model():
    vectors, identities = build_templates(get_all_students())
    return {'X': vectors, 'y': identities} if vectors else None

def train_classifier():
    # Refresh face templates only; keep expensive voice/separation models loaded.
    get_trained_model.clear()
    return bool(get_trained_model())

def predict_attendance(class_image_np):
    groups = get_face_embedding_groups(class_image_np)
    model = get_trained_model()
    if not model:
        return {}, [], len(groups)
    detected = {}
    for group in groups:
        matches = {match_face(encoding, model['X'], model['y'])[0] for encoding in group}
        matches.discard(None)
        # Conflicting identities between original and corrected versions require a retry.
        if len(matches) == 1:
            detected[matches.pop()] = True
    return detected, sorted(set(model['y'])), len(groups)
