import numpy as np
import pandas as pd
from digitize import Digitize

class SentenceProcessor:
    def __init__(self, filename=None, col_name="ENGLISH", padding=25):
        self.sentences = []
        self.raw_sentences = []  # Store original sentences
        self.padding = padding
        self.col_name = col_name
        if filename:
            self.load_sentences(filename) 

    def load_sentences(self, filename):
        df = pd.read_csv(filename)
        self.raw_sentences = df[f"{self.col_name}"].tolist()
        self.sentences = [
            self.pad_sentence(Digitize(sentence, self.padding).encode())
            for sentence in self.raw_sentences
        ]

    def pad_sentence(self, sentence):
        return sentence + [0] * (self.padding - len(sentence)) if len(sentence) < self.padding else sentence[: self.padding]

class ContinuousPatternRetriever:
    def __init__(self, sentences, raw_sentences, beta=8):
        self.sentences = sentences
        self.raw_sentences = raw_sentences
        self.beta = beta

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def continuous_update_rule(self, X, z):
        return X.T @ self.softmax(self.beta * X @ z)

    def retrieve_store_continuous(self, num_patterns, num_plot=10):
        X = np.array(self.sentences[:num_patterns])

        for j in range(num_plot):
            mask = np.ones(X.shape[1], dtype=int)
            mask[-(X.shape[1] // 2):] = 0  # Mask the last portion
            z = (X[j] * mask).reshape(-1, 1)
            out = self.continuous_update_rule(X, z)

            original_sentence = self.raw_sentences[j]
            tokenized_original = (X[j]).astype(int).tolist()
            masked = z.flatten()
            reconstruction = out.flatten()

            # Scale reconstruction back to the original range for decoding
            scaled_reconstruction = (reconstruction).astype(int).tolist()

            # Decode the reconstructed output
            digitizer = Digitize("", padding=len(tokenized_original))
            decoded_reconstruction = digitizer.decode(scaled_reconstruction)
                
            print("********************")
            print("Original Sentence:" ,original_sentence)
            print("Tokenized (Original):", tokenized_original)
            print("Masked:", masked)
            print("Reconstructed (tokenized):", scaled_reconstruction)
            print("Decoded Reconstruction:", decoded_reconstruction)

if __name__ == "__main__":
    processor = SentenceProcessor("newsents.csv", col_name="TEST", padding=50)
    num_patterns = 10  # Number of patterns to retrieve and store
    retriever = ContinuousPatternRetriever(processor.sentences, processor.raw_sentences)
    retriever.retrieve_store_continuous(num_patterns=num_patterns)

