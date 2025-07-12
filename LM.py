import re
import numpy as np
from collections import defaultdict
import string
import os
from sentence_transformers import SentenceTransformer

# Configure numpy to display arrays without excessive truncation
np.set_printoptions(threshold=1000)  # Adjusted for readability

# Step 1: Read and preprocess the .txt file
def load_and_preprocess(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist. Please check the path and file name.")
    if not file_path.lower().endswith('.txt'):
        raise ValueError(f"The file '{file_path}' is not a .txt file.")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().lower()
        if not text.strip():
            raise ValueError(f"The file '{file_path}' is empty or contains no readable text.")
        text = re.sub(f'[{string.punctuation}]', '', text)
        words = text.split()
        if not words:
            raise ValueError("No valid words found after preprocessing.")
        return words
    except Exception as e:
        raise Exception(f"Error processing file '{file_path}': {str(e)}")

# Step 2: Build vocabulary and assign unique IDs
def build_vocabulary(words):
    vocab = sorted(set(words))
    word_to_id = {word: idx for idx, word in enumerate(vocab)}
    id_to_word = {idx: word for idx, word in enumerate(vocab)}
    return vocab, word_to_id, id_to_word

# Step 3: Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)  # Pre-trained model for embeddings

# Step 4: Build co-occurrence matrix using Sentence Transformer embeddings
def build_cooccurrence_matrix(words, vocab, word_to_id, window_size=2):
    vocab_size = len(vocab)
    cooc_matrix = np.zeros((vocab_size, vocab_size))
    word_embeddings = model.encode(vocab, convert_to_numpy=True, show_progress_bar=False)
    
    for i in range(len(words)):
        word = words[i]
        if word not in word_to_id:
            continue
        word_id = word_to_id[word]
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        for j in range(start, end):
            if i != j and words[j] in word_to_id:
                context_id = word_to_id[words[j]]
                # Compute cosine similarity between word embeddings
                emb1 = word_embeddings[word_id]
                emb2 = word_embeddings[context_id]
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-9)
                cooc_matrix[word_id][context_id] += similarity
    return cooc_matrix

# Step 5: Get sentence embedding using Sentence Transformer
def get_sentence_embedding(sentence, word_to_id):
    sentence = re.sub(f'[{string.punctuation}]', '', sentence.lower())
    words = sentence.split()
    embeddings = []
    for word in words:
        if word in word_to_id:
            word_embedding = model.encode([word], convert_to_numpy=True, show_progress_bar=False)[0]
            embeddings.append(word_embedding)
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        print(f"Warning: No known words in sentence '{sentence}'. Returning zero vector.")
        return np.zeros(384)  # 384 is the embedding size for 'all-MiniLM-L6-v2'

# Step 6: Check if a sentence is logical
def is_logical_sentence(sentence, word_to_id, cooc_matrix, threshold=0.5):
    sentence = re.sub(f'[{string.punctuation}]', '', sentence.lower())
    words = sentence.split()
    if not words:
        print("Warning: Empty sentence provided.")
        return False
    total_score = 0
    count = 0
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        if word1 in word_to_id and word2 in word_to_id:
            id1, id2 = word_to_id[word1], word_to_id[word2]
            score = cooc_matrix[id1][id2]
            total_score += score
            count += 1
    if count == 0:
        print("Warning: No valid word pairs found in sentence.")
        return False
    avg_score = total_score / count
    return avg_score > threshold

# Main function
def main(file_path):
    try:
        words = load_and_preprocess(file_path)
    except Exception as e:
        print(e)
        return
    
    vocab, word_to_id, id_to_word = build_vocabulary(words)
    vocab_size = len(vocab)
    
    cooc_matrix = build_cooccurrence_matrix(words, vocab, word_to_id)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Sample word-to-ID mapping: {list(word_to_id.items())[:5]}")
    
    while True:
        sample_word = input("Enter a word to get its ID (or 'quit' to skip): ")
        if sample_word.lower() == 'quit':
            break
        if sample_word in word_to_id:
            print(f"Word: {sample_word}, ID: {word_to_id[sample_word]}")
            embedding = model.encode([sample_word], convert_to_numpy=True, show_progress_bar=False)[0]
            print(f"Sentence Transformer embedding (first 10 dims): {embedding[:10]}")
        else:
            print(f"Word '{sample_word}' not in vocabulary.")
    
    while True:
        try:
            sample_id = input("Enter an ID to get its word (or 'quit' to skip): ")
            if sample_id.lower() == 'quit':
                break
            sample_id = int(sample_id)
            if sample_id in id_to_word:
                word = id_to_word[sample_id]
                print(f"ID: {sample_id}, Word: {word}")
                embedding = model.encode([word], convert_to_numpy=True, show_progress_bar=False)[0]
                print(f"Sentence Transformer embedding (first 10 dims): {embedding[:10]}")
            else:
                print(f"ID {sample_id} not in vocabulary.")
        except ValueError:
            print("Please enter a valid integer ID or 'quit'.")
    
    while True:
        sentence = input("Enter a sentence to get its embedding and check if logical (or 'quit' to exit): ")
        if sentence.lower() == 'quit':
            break
        embedding = get_sentence_embedding(sentence, word_to_id)
        print(f"Sentence embedding (first 10 dims): {embedding[:10]}")
        is_logical = is_logical_sentence(sentence, word_to_id, cooc_matrix)
        print(f"Is the sentence logical? {is_logical}")

if __name__ == "__main__":
    file_path = r"D:\Ai\FUNDAMENTALS_r2.txt"
    main(file_path)