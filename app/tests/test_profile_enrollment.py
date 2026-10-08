import io
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import soundfile as sf
from src.pipelines import profile_enrollment as pe
from src.database import db
from src.pipelines.voice_pipeline import enroll_voice_sample
from src.pipelines import voice_pipeline as vp

class ProfileEnrollmentTests(unittest.TestCase):
    def setUp(self):
        self.photos=[]
        for color in ('red','green','blue'):
            output=io.BytesIO()
            Image.new('RGB',(80,80),color).save(output,format='PNG')
            self.photos.append(output.getvalue())
        self.front=np.zeros(128);self.front[0]=1.
        self.left=self.front.copy();self.left[1]=0.1
        self.right=self.front.copy();self.right[2]=0.1

    def test_missing_required_inputs_rejected_before_processing(self):
        with patch.object(pe,'prepare_face_profile') as faces,patch.object(pe,'enroll_voice_sample') as voice:
            for name,photos,audio in [('',self.photos,b'audio'),('Student',[None,*self.photos[1:]],b'audio'),('Student',self.photos,None)]:
                with self.assertRaises(ValueError):pe.prepare_registration(name,photos,audio,[])
            faces.assert_not_called();voice.assert_not_called()

    def test_valid_registration_has_three_faces_and_voice(self):
        with patch.object(pe,'get_face_embeddings',side_effect=[[self.front],[self.left],[self.right]]),patch.object(pe,'enroll_voice_sample',return_value=[1.]*256):
            name,faces,voice=pe.prepare_registration(' Student ',self.photos,b'audio',[])
        self.assertEqual(name,'Student');self.assertEqual(len(faces),3);self.assertEqual(len(voice),256)

    def test_duplicate_images_and_different_people_rejected(self):
        with self.assertRaises(ValueError):pe.prepare_face_profile([self.photos[0]]*3,[])
        other=np.ones(128)
        with patch.object(pe,'get_face_embeddings',side_effect=[[self.front],[other],[self.right]]):
            with self.assertRaises(ValueError):pe.prepare_face_profile(self.photos,[])

    def test_missing_or_multiple_faces_rejected(self):
        for result in ([],[self.front,self.left]):
            with patch.object(pe,'get_face_embeddings',return_value=result):
                with self.assertRaises(ValueError):pe.prepare_face_profile(self.photos,[])

    def test_update_cannot_replace_another_students_identity(self):
        roster=[{'student_id':1,'face_embedding':self.front.tolist()}]
        with patch.object(pe,'get_face_embeddings',return_value=[self.front]):
            with self.assertRaises(ValueError):pe.prepare_face_profile(self.photos,roster,student_id=2)
            with self.assertRaises(ValueError):pe.prepare_face_profile(self.photos,roster)

    def test_database_rejects_missing_voice_without_write(self):
        with patch.object(db,'supabase') as client:
            with self.assertRaises(ValueError):db.create_student('Student',[self.front.tolist()]*3,None)
            with self.assertRaises(ValueError):db.create_student('Student',self.front.tolist(),[1.]*256)
            client.table.assert_not_called()

    def test_single_side_view_collision_does_not_block_registration(self):
        known = self.front.copy()
        known[1] = .65
        roster = [{'student_id': 1, 'face_embedding': known.tolist()}]
        right = self.front.copy()
        right[1] = -.1
        with patch.object(pe, 'get_face_embeddings', side_effect=[[self.front], [self.left], [right]]):
            self.assertEqual(len(pe.prepare_face_profile(self.photos, roster)), 3)

    def test_two_views_matching_same_student_block_duplicate(self):
        known = self.front.copy()
        known[1] = .65
        right = self.front.copy()
        right[1] = .2
        roster = [{'student_id': 1, 'face_embedding': known.tolist()}]
        with patch.object(pe, 'get_face_embeddings', side_effect=[[self.front], [self.left], [right]]):
            with self.assertRaisesRegex(ValueError, 'two photos'):
                pe.prepare_face_profile(self.photos, roster)

    def test_ambiguous_face_matches_do_not_claim_duplicate_identity(self):
        roster = [{'student_id': 1, 'face_embedding': self.front.tolist()},
                  {'student_id': 2, 'face_embedding': self.front.tolist()}]
        with patch.object(pe, 'get_face_embeddings', side_effect=[[self.front], [self.left], [self.right]]):
            self.assertEqual(len(pe.prepare_face_profile(self.photos, roster)), 3)

    def test_short_and_silent_voice_recordings_rejected(self):
        for seconds,value in [(2,0.1),(10,0.)]:
            audio=io.BytesIO()
            sf.write(audio,np.full(seconds*16000,value,dtype=np.float32),16000,format='WAV')
            with self.assertRaises(ValueError):enroll_voice_sample(audio.getvalue())

    def test_usable_voice_required_after_silence_removal(self):
        audio=(0.1*np.sin(2*np.pi*200*np.arange(10*16000)/16000)).astype(np.float32)
        with patch.object(vp.librosa,'load',return_value=(audio,16000)),patch.object(vp,'preprocess_wav',return_value=audio[:16000]),patch.object(vp,'load_voice_encoder') as loader:
            with self.assertRaises(ValueError):enroll_voice_sample(b'recording')
            loader.assert_not_called()
        with patch.object(vp.librosa,'load',return_value=(audio,16000)),patch.object(vp,'preprocess_wav',return_value=audio),patch.object(vp,'load_voice_encoder') as loader:
            loader.return_value.embed_utterance.return_value=np.ones(256)
            self.assertEqual(len(enroll_voice_sample(b'recording')),256)

if __name__=='__main__':unittest.main()
