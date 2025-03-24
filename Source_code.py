import os
import pdfplumber
import docx2txt
import streamlit as st
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download("punkt")

# Streamlit UI
st.title("📄 AI Resume Screening and ranking System")
st.subheader("Upload Resumes")

# Upload resumes
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf", "docx"], accept_multiple_files=True)

# Job description input
job_description = st.text_area(
    "Enter Job Description",
    """Looking for a Data Scientist with expertise in Python, Machine Learning, and NLP. 
    Experience with TensorFlow, Scikit-learn, and data visualization is a plus.""",
)

# Function to extract text from resumes
def extract_text_from_file(file):
    text = ""
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    elif file.name.endswith(".docx"):
        text = docx2txt.process(file)
    return text.strip()

# Function to rank resumes
def rank_resumes(job_description, uploaded_files):
    resume_texts = []
    resume_names = []

    for file in uploaded_files:
        text = extract_text_from_file(file)
        if text:
            resume_texts.append(text)
            resume_names.append(file.name)

    if not resume_texts:
        st.warning("No valid resumes found!")
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    texts = [job_description] + resume_texts
    tfidf_matrix = vectorizer.fit_transform(texts)

    similarity_scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:]).flatten()
    ranked_resumes = sorted(zip(resume_names, similarity_scores), key=lambda x: x[1], reverse=True)

    return ranked_resumes

# Process resumes when button is clicked
if st.button("Rank Resumes"):
    if not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        ranked_results = rank_resumes(job_description, uploaded_files)

        if ranked_results:
            df = pd.DataFrame(ranked_results, columns=["Resume", "Score"])
            df.index += 1  # Start index from 1
            st.subheader("📊 Ranking Resumes")
            st.dataframe(df)
