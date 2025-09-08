import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="My Notes App",
    page_icon="📝",
    layout="centered"
)

st.markdown("""
<style>
    .note-box {
        background-color: rgba(240, 242, 246, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 4px solid #1f77b4;
    }
    .note-date {  
        font-size: 0.8rem;
        color: #888;
        margin-top: 5px;
    }
    div[data-testid="stForm"] {
        border: 1px solid rgba(230, 230, 230, 0.2);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    @media (prefers-color-scheme: dark) {
        .note-box {
            background-color: rgba(30, 30, 40, 0.5);
            color: #ffffff;
        }
    }
</style>
""", unsafe_allow_html=True)

st.title("📝 My Notes")

if "notes" not in st.session_state:
    st.session_state.notes = []

with st.form("note_form"):
    note_text = st.text_area("Write your note here:", height=100)
    add_note = st.form_submit_button("Add Note",type="primary")

    if add_note and note_text.strip():
        note_data = {
            "text": note_text.strip(),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        st.session_state.notes.append(note_data)
        st.success("Note added successfully!")
        st.rerun()

if st.session_state.notes:
    st.subheader("Your Notes")
    
    for note in reversed(st.session_state.notes):
        st.markdown(f'<div class="note-box">{note["text"]}<div class="note-date">Added: {note["time"]}</div></div>', unsafe_allow_html=True)
    
    if st.button("Clear All Notes",type="primary"):
        st.session_state.notes = []
        st.rerun()
else:
    st.info("No notes yet. Add your first note above!")