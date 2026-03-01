from flask import Flask, render_template, request
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import webbrowser
import threading

app = Flask(__name__)

# Load CSV file
df = pd.read_csv("faq.csv")

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Clean stored questions
df['Cleaned_Question'] = df['Question'].apply(clean_text)

# Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
question_vectors = vectorizer.fit_transform(df['Cleaned_Question'])

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    category = ""

    if request.method == "POST":
        user_question = request.form.get("question", "")
        cleaned_user = clean_text(user_question)

        if cleaned_user == "":
            answer = "Please enter a question."
        else:
            user_vector = vectorizer.transform([cleaned_user])
            similarity_scores = cosine_similarity(user_vector, question_vectors)

            max_score = similarity_scores.max()
            max_index = similarity_scores.argmax()

            if max_score >= 0.70:
                answer = df.iloc[max_index]["Answer"]
                category = df.iloc[max_index]["Category"]
            else:
                answer = "Not Found"
                category = "No Matching Category"

    return render_template("index.html", answer=answer, category=category)

# Function to open browser automatically
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    # Open the browser in a new thread after 1 second
    threading.Timer(1, open_browser).start()
    # Run Flask app
    app.run(host='127.0.0.1', port=5000, debug=True)