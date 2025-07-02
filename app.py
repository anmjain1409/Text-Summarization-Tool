import streamlit as st
from transformers import pipeline
import language_tool_python
from language_tool_python.utils import RateLimitError

# Load summarizer model
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()

# Use LanguageTool Public API
def get_grammar_issues(text):
    try:
        tool = language_tool_python.LanguageToolPublicAPI('en-US')
        matches = tool.check(text)
        return [match.message for match in matches]
    except RateLimitError:
        return "RATE_LIMIT"

st.set_page_config(page_title="Text Summarizer & Grammar Checker", layout="centered")
st.title("📝 Text Summarizer with Grammar Check")

text_input = st.text_area("📥 Paste your text here", height=300)

if st.button("Analyze"):
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("🔍 Analyzing..."):

            # Word count
            word_count = len(text_input.split())

            # Grammar check
            grammar_issues = get_grammar_issues(text_input)

            # Summarization
            summary = summarizer(text_input, max_length=120, min_length=30, do_sample=False)[0]['summary_text']

        st.subheader("📌 Summary")
        st.write(summary)

        st.subheader("🔢 Word Count")
        st.write(word_count)

        st.subheader("🧠 Grammar Issues")
        if grammar_issues == "RATE_LIMIT":
            st.error("⚠️ Rate limit reached for LanguageTool public API. Try again later.")
        elif grammar_issues:
            for issue in grammar_issues:
                st.write(f"🔸 {issue}")
        else:
            st.success("No grammar issues found!")
