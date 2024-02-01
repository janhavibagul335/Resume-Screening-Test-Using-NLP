from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def preprocess_text(text):
    return text.lower()

def is_match(resume, job_description):
    job_description = preprocess_text(job_description)
    vectorizer = CountVectorizer()
    resume_matrix = vectorizer.fit_transform([preprocess_text(resume), job_description])
    similarity_matrix = cosine_similarity(resume_matrix)
    similarity_score = similarity_matrix[0, 1]
    
    # You can adjust the threshold as needed
    match_threshold = 0.5
    return similarity_score >= match_threshold

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume = request.form['resume']
        match_result = is_match(resume, job_description)
        return render_template('result.html', job_description=job_description, resume=resume, match_result=match_result)

if __name__ == '__main__':
    app.run(debug=True)
