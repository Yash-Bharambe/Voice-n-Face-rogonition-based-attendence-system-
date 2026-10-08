"""Photometric correction only: no synthesis of missing facial details."""
import cv2
import numpy as np

def lighting_info(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not image.size:
        raise ValueError('Provide a nonempty RGB image with 8-bit pixels.')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    median = float(np.median(gray))
    spread = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    level = 'dark' if median < 65 else 'bright' if median > 190 else 'low contrast' if spread < 45 else 'normal'
    return {'level': level, 'median': median, 'spread': spread,
            'usable': int(gray.max()) - int(gray.min()) >= 6,
            'shadow_clipping': float(np.mean(np.all(image <= 5, axis=2))),
            'highlight_clipping': float(np.mean(np.all(image >= 250, axis=2)))}

def normalize_lighting(image):
    info = lighting_info(image)
    if not info['usable'] or info['level'] == 'normal':
        return np.ascontiguousarray(image.copy()), info
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    light = lab[:, :, 0]
    median = max(1., min(254., float(np.median(light))))
    gamma = float(np.clip(np.log(125/255) / np.log(median/255), 0.45, 2.0))
    table = np.clip(255 * (np.arange(256)/255) ** gamma, 0, 255).astype(np.uint8)
    adjusted = cv2.LUT(light, table)
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(adjusted)
    lab[:, :, 0] = cv2.addWeighted(adjusted, 0.65, contrast, 0.35, 0)
    return np.ascontiguousarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)), info

def usable_face_crop(crop):
    info = lighting_info(crop)
    return info['usable'] and info['spread'] >= 8 and max(info['shadow_clipping'], info['highlight_clipping']) < 0.7
