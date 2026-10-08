import streamlit as st
from src.database.config import supabase
from src.database.db import get_all_students, create_student
from src.pipelines.face_pipeline import get_trained_model
from src.pipelines.profile_enrollment import ANGLES, prepare_face_profile, prepare_registration
from src.pipelines.voice_service import enroll_voice_sample
from src.pipelines.voice_challenge import get_rolls, save_rolls

def capture_three_photos(prefix):
    source = st.radio('Photo source', ['Camera', 'Upload photos'], key=f'{prefix}_source')
    st.caption('Capture three different views of the same person. Face the camera for Front, then turn slightly left and right (about 30–60 degrees). Keep your face visible and use even lighting.')
    photos = []
    for angle in ANGLES:
        key = f'{prefix}_{source}_{angle}'
        if source == 'Camera':
            photo = st.camera_input(f'{angle} photo (required)', key=key)
        else:
            photo = st.file_uploader(f'{angle} photo (required)', type=['jpg','jpeg','png'], key=key)
        photos.append(photo.getvalue() if photo is not None else None)
    return photos

def registration_form():
    st.header('Register student')
    st.write('All three face photos and a voice sample are required.')
    name = st.text_input('Your name', key='register_name')
    photos = capture_three_photos('register')
    st.caption('Voice sample: record 8–30 seconds. Say your name, roll number, and a few sentences in a quiet room.')
    audio = st.audio_input('Voice sample (required)', key='register_voice')
    ready = bool(name.strip() and all(photos) and audio is not None)
    if st.button('Create Account', type='primary', disabled=not ready):
        try:
            with st.spinner('Checking photos and enrolling your voice…'):
                name, faces, voice = prepare_registration(name, photos, audio.getvalue(), get_all_students())
                response = create_student(name, face_embedding=faces, voice_embedding=voice)
                if not response:
                    raise ValueError('Your profile was not saved. Try again.')
            st.session_state.student_data = response[0]
            st.session_state.is_logged_in = True
            st.session_state.user_role = 'student'
            get_trained_model.clear()
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error('Could not complete registration. Check your connection and try again.')

def profile_settings(student_data):
    sid = student_data['student_id']
    revision = st.session_state.get('profile_revision', 0)
    with st.expander('Update face images'):
        st.caption('Replace your saved face references with a new front, left-side, and right-side set. Your current references stay in place until all three pass validation and save successfully.')
        photos = capture_three_photos(f'update_faces_{sid}_{revision}')
        if st.button('Update face images', disabled=not all(photos)):
            try:
                with st.spinner('Checking the new face photos…'):
                    references = prepare_face_profile(photos, get_all_students(), student_id=sid)
                    response = supabase.table('students').update({'face_embedding': references}).eq('student_id', sid).execute()
                    if not response.data:
                        raise ValueError('Your face references were not saved. Try again.')
                st.session_state.student_data['face_embedding'] = references
                get_trained_model.clear()
                st.success('All three face images updated.')
            except ValueError as exc:
                st.error(str(exc))
            except Exception:
                st.error('Could not update face images. Try again.')
    with st.expander('Update voice sample'):
        st.caption('Record 8–30 seconds using the same microphone and distance you use for attendance. Say your name and repeat a practice phrase like: Roll number two three, code apple river silver garden. Speak naturally. Your saved voice is replaced only after validation and a successful save.')
        audio = st.audio_input('New voice sample', key=f'update_voice_{sid}_{revision}')
        if st.button('Update voice sample', disabled=audio is None):
            try:
                with st.spinner('Checking and saving your voice…'):
                    voice = enroll_voice_sample(audio.getvalue())
                    response = supabase.table('students').update({'voice_embedding': voice}).eq('student_id', sid).execute()
                    if not response.data:
                        raise ValueError('Your voice sample was not saved. Try again.')
                st.session_state.student_data['voice_embedding'] = voice
                st.success('Voice sample updated.')
            except ValueError as exc:
                st.error(str(exc))
            except Exception:
                st.error('Could not update the voice sample. Try again.')
    with st.expander('Voice attendance roll number'):
        roll = st.text_input('Your roll number (digits)', value=get_rolls().get(sid,''), key=f'profile_roll_{sid}')
        if st.button('Save roll number'):
            try:
                if not roll.strip():
                    raise ValueError('Enter your roll number.')
                save_rolls({sid: roll})
                st.success('Roll number saved.')
            except ValueError as exc:
                st.error(str(exc))
