import unittest
from unittest.mock import patch
import cv2
import numpy as np
from src.pipelines.face_lighting import lighting_info, normalize_lighting, usable_face_crop

class LightingTests(unittest.TestCase):
    def gradient(self, low, high):
        gray = np.tile(np.linspace(low, high, 160, dtype=np.uint8), (120,1))
        return np.repeat(gray[:,:,None],3,axis=2)

    def test_dark_and_bright_detail_preserved_and_brightness_improved(self):
        for low, high, direction in [(10,60,1),(200,240,-1)]:
            image = self.gradient(low,high)
            corrected, info = normalize_lighting(image)
            self.assertEqual(corrected.shape,image.shape)
            self.assertEqual(corrected.dtype,np.uint8)
            old = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).mean()
            new = cv2.cvtColor(corrected,cv2.COLOR_RGB2GRAY).mean()
            self.assertGreater(direction*(new-old),0)
            self.assertTrue(info['usable'])

    def test_normal_lighting_unchanged(self):
        image = self.gradient(50,200)
        corrected,info=normalize_lighting(image)
        self.assertEqual(info['level'],'normal')
        np.testing.assert_array_equal(corrected,image)

    def test_all_black_and_white_not_reconstructed(self):
        for value in (0,255):
            image=np.full((120,160,3),value,dtype=np.uint8)
            self.assertFalse(lighting_info(image)['usable'])
            self.assertFalse(usable_face_crop(image))
            np.testing.assert_array_equal(normalize_lighting(image)[0],image)

    def test_clipped_face_rejected(self):
        image=self.gradient(20,180)
        image[:100,:,:]=255
        self.assertFalse(usable_face_crop(image))

    def test_different_identity_from_correction_rejected(self):
        from src.pipelines import face_pipeline as fp
        a=np.zeros(128);a[0]=1
        b=np.zeros(128);b[1]=1
        model={'X':[a,b],'y':[1,2]}
        with patch.object(fp,'get_face_embedding_groups',return_value=[[a,b]]),patch.object(fp,'get_trained_model',return_value=model):
            detected,_,count=fp.predict_attendance(self.gradient(10,60))
        self.assertEqual(detected,{})
        self.assertEqual(count,1)

    def test_original_and_corrected_same_face_count_once(self):
        from src.pipelines import face_pipeline as fp
        a=np.zeros(128);a[0]=1
        model={'X':[a],'y':[1]}
        with patch.object(fp,'get_face_embedding_groups',return_value=[[a,a]]),patch.object(fp,'get_trained_model',return_value=model):
            detected,_,count=fp.predict_attendance(self.gradient(10,60))
        self.assertEqual(detected,{1:True})
        self.assertEqual(count,1)

if __name__=='__main__':
    unittest.main()
