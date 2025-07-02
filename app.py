import streamlit as st
from transformers import pipeline

# Load summarizer model (cached for performance)
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()

st.set_page_config(page_title="Text Summarizer", layout="centered")
st.title("📝 Text Summarizer")

text_input = st.text_area("📥 Paste your text here", height=300)

if st.button("Summarize"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarizer(text_input, max_length=120, min_length=30, do_sample=False)[0]['summary_text']
            word_count = len(text_input.split())

        st.subheader("📌 Summary")
        st.write(summary)

        st.subheader("🔢 Word Count")
        st.write(word_count)
