import os

import streamlit as st


from supabase import create_client, Client

def get_setting(name):
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except (FileNotFoundError, KeyError):
        st.error(f"Configure {name} in your hosting environment or Streamlit secrets.")
        st.stop()


supabase: Client = create_client(get_setting("SUPABASE_URL"), get_setting("SUPABASE_KEY"))
