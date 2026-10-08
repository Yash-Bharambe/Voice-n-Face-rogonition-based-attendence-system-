import unittest
import numpy as np
from unittest.mock import patch, Mock
from src.pipelines.face_matching import face_templates, build_templates, match_face, merge_boxes

class FaceMatchingTests(unittest.TestCase):
    def setUp(self):
        self.front = np.zeros(128)
        self.front[0] = 1
        self.side = self.front.copy()
        self.side[1] = 0.8
        self.other = self.front.copy()
        self.other[2] = 1.2

    def test_old_and_multi_angle_profiles(self):
        vectors, ids = build_templates([
            {'student_id': 1, 'face_embedding': [self.front.tolist(), self.side.tolist()]},
            {'student_id': 2, 'face_embedding': self.other.tolist()}])
        self.assertEqual(ids, [1, 1, 2])
        self.assertEqual(match_face(self.side, vectors, ids)[0], 1)
        self.assertEqual(match_face(self.front, vectors, ids)[0], 1)

    def test_unknown_side_face_is_not_automatically_present(self):
        self.assertIsNone(match_face(self.side, [self.front], [1])[0])

    def test_ambiguous_identity_rejected_but_same_person_templates_allowed(self):
        near = self.front.copy()
        near[1] = 0.01
        self.assertIsNone(match_face(self.front, [self.front, near], [1, 2])[0])
        self.assertEqual(match_face(self.front, [self.front, near], [1, 1])[0], 1)

    def test_invalid_templates_ignored(self):
        for value in (None, [], [1,2], [float('nan')]*128, [0]*128):
            self.assertEqual(face_templates(value), [])

    def test_duplicate_detection_boxes_merge(self):
        boxes = [(10,10,100,100), (15,15,95,95), (150,10,240,100)]
        self.assertEqual(merge_boxes(boxes), [boxes[0], boxes[2]])

    def test_mirrored_profile_detections_map_back_to_original(self):
        from src.pipelines import face_pipeline as fp
        yunet = Mock()
        yunet.detect.return_value = (None, None)
        profile = Mock()
        profile.detectMultiScale.return_value = [(5, 5, 30, 30)]
        with patch.object(fp, 'load_pose_detectors', return_value=(yunet, profile)), \
             patch.object(fp, 'load_dlib_models', return_value=(lambda image, upscale: [], None, None)):
            boxes = fp.detect_face_boxes(np.zeros((80, 120, 3), dtype=np.uint8))
        self.assertEqual(boxes, [(5, 5, 35, 35), (85, 5, 115, 35)])

    def test_side_reference_marks_student_present(self):
        from src.pipelines import face_pipeline as fp
        model = {'X': [self.front, self.side, self.other], 'y': [1, 1, 2]}
        with patch.object(fp, 'get_face_embedding_groups', return_value=[[self.side]]), \
             patch.object(fp, 'get_trained_model', return_value=model):
            detected, identities, count = fp.predict_attendance(np.zeros((80, 120, 3), dtype=np.uint8))
        self.assertEqual(detected, {1: True})
        self.assertEqual(identities, [1, 2])
        self.assertEqual(count, 1)

if __name__ == '__main__':
    unittest.main()
