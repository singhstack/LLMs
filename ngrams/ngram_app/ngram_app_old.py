from flask import Flask, render_template, request
from ngram_models_utils import create_ngrams, generate_sentence, get_probability, preprocess_new, probability_helper, predict
import pickle
from collections import Counter

app  = Flask(__name__)
# Load the data from the pickle file

with open('file_text.pkl', 'rb') as pickle_file:
    data = pickle.load(pickle_file)

austen = data['blake-poems.txt']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Replace 'input1', 'input2', etc. with your actual input field names
    initial_sequence = request.form['initial_sequence']
    n_grams = request.form['n_grams']
    sentence_length = request.form['sentence_length']
    n_grams, sentence_length = int(n_grams), int(sentence_length)
    probs_austen  = get_probability(austen,n_grams ,type = "smooth")

    # Here, call your function with the inputs
    output_sequence = generate_sentence(probs_austen,initial_sequence, n_grams, sentence_length)  # Replace with your actual function

    return render_template('result.html', sequence=output_sequence)

if __name__ == '__main__':
    app.run(debug=True)
