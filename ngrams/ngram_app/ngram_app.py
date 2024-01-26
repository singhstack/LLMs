from flask import Flask, render_template, request,jsonify
from ngram_models_utils import generate_sentence, get_probability, preprocess_new, probability_helper, predict,create_ngrams
import pickle
from collections import Counter

app  = Flask(__name__)
# Load the data from the pickle file

with open('file_text.pkl', 'rb') as pickle_file:
    data = pickle.load(pickle_file)

blake = data['blake-poems.txt']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_sequence', methods=['POST'])

def generate_sequence():
    # Replace 'input1', 'input2', etc. with your actual input field names
    initial_sequence = request.form['initial_sequence']
    n_grams = request.form['n_grams']
    sentence_length = request.form['sentence_length']
    n_grams, sentence_length = int(n_grams), int(sentence_length)
    probs_blake  = get_probability(blake,n_grams ,type = "smooth")

    # Here, call your function with the inputs
    output_sequence = generate_sentence(probs_blake,initial_sequence, n_grams, sentence_length) 

    #return render_template('result.html', sequence=output_sequence)
    return jsonify({'sequence': output_sequence})

if __name__ == '__main__':
    app.run(debug=True)
