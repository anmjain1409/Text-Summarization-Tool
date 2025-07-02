import streamlit as st
from transformers import pipeline
import language_tool_python

# Load summarization model (cached for performance)
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()

# Use LanguageTool Public API (Java not required)
tool = language_tool_python.LanguageToolPublicAPI('en-US')

st.set_page_config(page_title="Text Summarizer & Grammar Checker", layout="centered")
st.title("📝 Text Summarizer with Grammar Check")

text_input = st.text_area("📥 Paste your text here", height=300)

if st.button("Analyze"):
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("🔍 Analyzing..."):

            # Grammar check
            matches = tool.check(text_input)
            grammar_issues = [match.message for match in matches]

            # Word count
            word_count = len(text_input.split())

            # Summarization
            summary = summarizer(text_input, max_length=120, min_length=30, do_sample=False)[0]['summary_text']

        st.subheader("📌 Summary")
        st.write(summary)

        st.subheader("🔢 Word Count")
        st.write(word_count)

        st.subheader("🧠 Grammar Issues")
        if grammar_issues:
            for issue in grammar_issues:
                st.write(f"🔸 {issue}")
        else:
            st.success("No grammar issues found!")
