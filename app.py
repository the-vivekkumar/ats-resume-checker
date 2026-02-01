from flask import Flask, render_template, request
import os
import re
from docx import Document
import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# -------- TEXT EXTRACTORS --------

def extract_text_from_pdf(path):
    text = ""
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    return text


# -------- ATS SCORING LOGIC --------

def calculate_score(resume_text, jd_text):
    resume_text = clean_text(resume_text)
    jd_text = clean_text(jd_text)

    vectorizer = CountVectorizer().fit([resume_text, jd_text])
    vectors = vectorizer.transform([resume_text, jd_text]).toarray()

    resume_vector = vectors[0]
    jd_vector = vectors[1]

    matched = []
    missing = []

    features = vectorizer.get_feature_names_out()

    for i, word in enumerate(features):
        if jd_vector[i] > 0:
            if resume_vector[i] > 0:
                matched.append(word)
            else:
                missing.append(word)

    score = int((len(matched) / (len(matched) + len(missing))) * 100) if (matched or missing) else 0

    return score, matched[:20], missing[:20]


# -------- ROUTES --------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["resume"]
        jd_text = request.form["jd"]

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        if file.filename.endswith(".pdf"):
            resume_text = extract_text_from_pdf(path)
        else:
            resume_text = extract_text_from_docx(path)

        score, matched, missing = calculate_score(resume_text, jd_text)

        return render_template(
            "index.html",
            score=score,
            matched=matched,
            missing=missing
        )

    return render_template("index.html", score=None)


if __name__ == "__main__":
    app.run(debug=True)
