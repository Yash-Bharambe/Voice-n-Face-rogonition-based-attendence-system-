import hashlib
from datetime import datetime
import pandas as pd
import streamlit as st
from src.database.config import supabase
from src.database.db import create_attendance
from src.pipelines.voice_service import identify_voices
from src.pipelines.voice_challenge import get_rolls, save_rolls, INDIA


@st.dialog('Voice Attendance', width='large')
def voice_attendance_dialog(selected_subject_id):
    enrolled = supabase.table('subject_students').select('*, students(*)').eq('subject_id', selected_subject_id).execute().data
    students = [node['students'] for node in enrolled if node.get('students')]
    if not students:
        st.warning('No students are enrolled in this course.')
        return
    day = datetime.now(INDIA).date().isoformat()
    rolls = get_rolls()
    profiles = tuple((s['student_id'], repr(s.get('voice_embedding'))) for s in students)
    context = ('voice_only', selected_subject_id, day, tuple(sorted(rolls.items())), profiles)
    if st.session_state.get('voice_context') != context:
        st.session_state.voice_context = context
        st.session_state.voice_verified = {}
        st.session_state.voice_used_audio = set()
        st.session_state.voice_capture = st.session_state.get('voice_capture', 0) + 1
    st.write('Speak any clear sentence. Attendance is identified from your enrolled voice.')
    mode = st.radio('Recording mode', ['One student', 'Overlapping group (experimental)'])
    overlap = mode == 'Overlapping group (experimental)'
    speaker_count = st.selectbox('Students speaking at the same time', [2, 3]) if overlap else 1
    st.caption('Record 3–12 seconds for a group, or 3–20 seconds for one student.')
    st.caption('Use the same microphone as enrollment, about 20–40 cm from your mouth. Speak clearly at a normal volume. Quiet audio is adjusted automatically.')
    if overlap:
        st.info('Use exactly the selected number of speakers. Separation may fail in classroom noise. Four or five overlapping speakers are not supported; split into smaller groups.')
    st.caption('Use live recordings. This check cannot detect edited recordings or synthesized voices.')
    with st.expander('Set student roll numbers'):
        st.caption('Roll numbers are stored on this computer. Use digits; leading zeros are ignored.')
        roster = pd.DataFrame([{'Student ID': s['student_id'], 'Name': s['name'], 'Roll number': rolls.get(s['student_id'], '')} for s in students])
        edited = st.data_editor(roster, disabled=['Student ID', 'Name'], hide_index=True,
                                key=f'voice_rolls_{selected_subject_id}', width='stretch')
        if st.button('Save roll numbers'):
            try:
                save_rolls(dict(zip(edited['Student ID'], edited['Roll number'])))
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    candidates = {str(s['student_id']): s for s in students if s.get('voice_embedding')}
    if not candidates:
        st.info('Students need enrolled voice samples before recording attendance.')
        return
    missing = [s['name'] for s in students if not s.get('voice_embedding')]
    if missing:
        st.warning('Missing voice enrollment: ' + ', '.join(missing))
    audio = st.audio_input('Record students speaking', key=f"voice_audio_{st.session_state.voice_capture}_{speaker_count}")
    if st.button('Verify group' if overlap else 'Verify student', type='primary', disabled=audio is None, width='stretch'):
        try:
            audio_bytes = audio.getvalue()
            digest = hashlib.sha256(audio_bytes).hexdigest()
            if digest in st.session_state.voice_used_audio:
                raise ValueError('This recording has already been verified. Record the next student.')
            with st.spinner('Comparing voices with enrolled students...'):
                if overlap:
                    stream_results = identify_voices(audio_bytes, candidates, speaker_count)
                    st.dataframe(pd.DataFrame([{k: r[k] for k in ('Stream', 'Status', 'Reason')} for r in stream_results]), hide_index=True, width='stretch')
                    accepted = [r for r in stream_results if r['Status'] == 'Verified']
                else:
                    accepted = [identify_voices(audio_bytes, candidates)]
                for result in accepted:
                    st.session_state.voice_verified[result['student_id']] = {'roll': rolls.get(result['student_id'], ''), 'score': result['score']}
                if accepted:
                    st.session_state.voice_used_audio.add(digest)
            if accepted:
                st.success(f"Verified {len(accepted)} student(s). Retry any remaining students separately.")
            else:
                st.warning('No students passed all checks. Record again or use one-student mode.')
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error('Audio analysis failed. Check the local server log and try again; no attendance was added.')
            import logging
            logging.getLogger(__name__).exception('Voice verification failed')
    verified = st.session_state.voice_verified
    if verified:
        if st.button('Record next student'):
            st.session_state.voice_capture += 1
            st.rerun()
        st.dataframe(pd.DataFrame([{'Name': s['name'], 'Roll number': verified[s['student_id']]['roll'],
            'Voice similarity': round(verified[s['student_id']]['score'], 3), 'Status': 'Verified present'}
            for s in students if s['student_id'] in verified]), hide_index=True, width='stretch')
        st.caption('Only verified students will be saved as present. Other students remain pending.')
        if st.button('Save verified attendance', type='primary', width='stretch'):
            if datetime.now(INDIA).date().isoformat() != day or get_rolls() != rolls:
                st.error('The date or roll numbers changed. Verify students again.')
                return
            timestamp = datetime.now(INDIA).isoformat()
            logs = [{'student_id': sid, 'subject_id': selected_subject_id, 'timestamp': timestamp, 'is_present': True} for sid in verified]
            try:
                create_attendance(logs)
            except Exception:
                st.error('Saving failed. Your verified results are still available; retry saving.')
                return
            st.session_state.voice_verified = {}
            st.session_state.voice_capture += 1
            st.success('Verified attendance saved.')
