from sentence_transformers import SentenceTransformer
import re

def get_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

def get_sentences_from_text(text):
    return re.split(r'[.:!?\n]+', text)

def get_embeddings_from_file(model, file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    sentences = get_sentences_from_text(text)
    return model.encode(sentences)

def get_embeddings_from_sentences(model, sentences):
    return model.encode(sentences)
