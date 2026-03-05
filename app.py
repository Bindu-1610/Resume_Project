from flask import Flask, render_template, request
import pickle
import re
import PyPDF2

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = re.sub(r'http\S+', '', str(text))
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    return text

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["resume"]

    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file)
    else:
        text = file.read().decode("utf-8")

    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    return render_template("index.html", prediction_text=prediction)

if __name__ == "_main_":
    app.run(debug=True)