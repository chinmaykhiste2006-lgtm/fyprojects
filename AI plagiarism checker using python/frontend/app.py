import streamlit as st
import json
from plag4 import analyzeText

st.set_page_config(page_title="AI Content & Plagiarism Detector", layout="centered")
st.title("🕵️‍♂️ AI Content & Plagiarism Detector")

option = st.radio("Choose input type:", ["Text Input", "Upload File"])
input_text = ""

if option == "Text Input":
    input_text = st.text_area("Enter your text here:", height=200)
elif option == "Upload File":
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
    if uploaded_file:
        input_text = uploaded_file.read().decode("utf-8")
        st.text_area("File Content", input_text, height=200)

if st.button("Analyze"):
    if input_text.strip():
        with st.spinner("Analyzing..."):
            results, aiPercent, verdict = analyzeText(input_text)

        totalPerplexity = sum(r['perplexity'] for r in results)
        totalEmotion = sum(r['emotionStrength'] for r in results)
        totalReadability = sum(r['readability'] for r in results)
        totalRichness = sum(r['lexicalRichness'] for r in results)
        totalSentLength = sum(r['sentenceLength'] for r in results)
        totalGrammarIssues = sum(r['grammarIssues'] for r in results)
        num_sentences = len(results)

        avgPerplexity = totalPerplexity / num_sentences
        avgEmotion = totalEmotion / num_sentences
        avgReadability = totalReadability / num_sentences
        avgRichness = totalRichness / num_sentences
        avgSentLength = totalSentLength / num_sentences
        avgGrammarIssues = totalGrammarIssues / num_sentences

        aiCount = sum(1 for r in results if r['isAi'])
        humanCount = num_sentences - aiCount

        st.subheader("📊 Averages & Summary")
        st.write(f"**Average Perplexity:** {avgPerplexity:.2f}")
        st.write(f"**Average Emotion Strength:** {avgEmotion:.2f}")
        st.write(f"**Average Readability (Flesch):** {avgReadability:.2f}")
        st.write(f"**Average Lexical Richness (TTR):** {avgRichness:.2f}")
        st.write(f"**Average Sentence Length:** {avgSentLength:.2f} words")
        st.write(f"**Average Grammar Issues per Sentence:** {avgGrammarIssues:.2f}")

        st.markdown("---")
        st.subheader("🧾 Verdict")
        st.write(f"**AI Sentences:** {aiCount}")
        st.write(f"**Human Sentences:** {humanCount}")
        st.write(f"**AI Content Percentage:** {aiPercent:.2f}%")
        st.success(f"**Final Verdict: {verdict}**")

        st.markdown("---")
        st.subheader("🔍 Sentence-Level Analysis")
        for i, r in enumerate(results, 1):
            with st.expander(f"Sentence {i}: {r['sentence']}"):
                st.json(r)

        st.download_button("📥 Download JSON Report", json.dumps(results, indent=2), file_name="analysis_report.json")
    else:
        st.warning("⚠️ Please provide text or upload a file.")
