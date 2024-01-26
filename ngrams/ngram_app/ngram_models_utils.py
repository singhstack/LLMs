import pandas as pd
import numpy as np
import nltk, string
from nltk.corpus import gutenberg
from collections import Counter
from nltk.tokenize import word_tokenize
import random, string

def preprocess_new(text):
    text = ' '.join(gutenberg.raw('blake-poems.txt').split())
    # Remove punctuation except for commas
    punctuation_to_remove = string.punctuation.replace(',', '')  # Keep commas
    translator = str.maketrans('', '', punctuation_to_remove)
    text = text.translate(translator)
    #text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize and lower case
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    return tokens

def create_ngrams(tokens, n):
    n_gram_tokens = []
    for i in range(len(tokens)-n):
        n_gram_tokens.append(tuple(tokens[i:i+n]))
    return n_gram_tokens

def probability_helper(sample,n):
    """
    sample: text sample
    n: n-gram size
    return: dataframe with probability
    """
    #get ngrams
    ngrams_sample = create_ngrams(sample,n)

    #get frequency
    ngram_frequency = Counter([tuple(ngram) for ngram in ngrams_sample])

    #ger probability
    df = pd.DataFrame.from_dict(ngram_frequency, orient='index').reset_index()
    df.columns = ['sequence',  'count']

    #convert first column into 2 columns where first column has n-1 words, the second column has nth word
    df['nth_word'] = df['sequence'].apply(lambda x: x[-1])

    def get_sequence(tuple):
        x = ''
        for i in range(len(tuple)-1):
            x+=(tuple[i])
            x+=','
        x = x[:-1]
        x = x.replace(","," ")
        return x

    df['sequence'] = df['sequence'].apply(lambda x: get_sequence(x))

    #get ids for sequences and predictions
    df_sorted = df.sort_values(by='sequence')
    df_sorted['sequence_id'] = range(1, len(df_sorted) + 1)
    df_new = df_sorted
    df_sorted = df_new.sort_values(by='nth_word')
    df_sorted['prediction_id'] = range(1, len(df_sorted) + 1)

    return df, df_sorted

def get_probability(sample,n,type = None):
    if type==None:

        df, df_sorted = probability_helper(sample,n)
        totals = df.groupby('sequence')['count'].sum().reset_index().rename(columns={'count':'total'})
        df_sorted = df_sorted.merge(totals, how = 'left', on = 'sequence')
        df_sorted['probability'] = df_sorted['count']/df_sorted['total']
    elif type =="smooth":
        df, df_sorted = probability_helper(sample,n)
        v = df_sorted['prediction_id'].max()
        
        totals = df.groupby('sequence')['count'].sum().reset_index().rename(columns={'count':'total'})
        df_sorted = df_sorted.merge(totals, how = 'left', on = 'sequence')
        df_sorted['probability'] = (df_sorted['count']+1)/(df_sorted['total'] + v)

    return df_sorted

def predict(data, sequence):
    """this function generates predictions based on probabilities seen in the dataset"""
    try:
        subset = data[data['sequence']==sequence.strip()]
        result = subset.iloc[subset['probability'].argmax()]['nth_word'] #return the word with max probability
        #print("sequence detected")
        return result
    except:
        result = random.choice(data['nth_word'].unique())
        #print("sequence not detected")
        return result

def generate_sentence(data, sequence, n,len ):
    """
    data: result of get_probability()
    sequence: should be n-1 words together
    len: number of predictions to be made
    """
    sentence = sequence
    sentence = sentence.strip()
    for i in range(len):
        n_minus_1_sequence = ' '.join(sentence.split(" ")[-n+1:])
        #print(f'sequence number {i+1}: {n_minus_1_sequence}')
        next_word = predict(data, n_minus_1_sequence)
        if next_word!=',':
            sentence = sentence + ' ' + next_word
        else:
            sentence+=next_word
    return sentence


'''
files = gutenberg.fileids()
text = [gutenberg.raw(fileid) for fileid in gutenberg.fileids()]
file_text = dict(zip(files, text))

for key, value in file_text.items():
    file_text[key] = preprocess_new(value)'''
