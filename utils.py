import pickle
import numpy as np


def get_embedding(word_dict, embedding_path, embedding_dim=300):
    # find existing word embeddings
    word_vec = {}
    with open(embedding_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))

    print('Found {0}/{1} words with embedding vectors'.format(
        len(word_vec), len(word_dict)))
    missing_word_num = len(word_dict) - len(word_vec)
    missing_ratio = round(float(missing_word_num) / len(word_dict), 4) * 100
    print('Missing Ratio: {}%'.format(missing_ratio))

    # handling unknown embeddings
    for word in word_dict:
        if word not in word_vec:
            # If word not in word_vec, create a random embedding for it
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            word_vec[word] = new_embedding
    print ("Filled missing words' embeddings.")
    print ("Embedding Matrix Size: ", len(word_vec))

    return word_vec

def save_embed(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print ('Embedding saved')

def load_embed(path):
    with open(path, 'rb') as f:
        return pickle.load(f)