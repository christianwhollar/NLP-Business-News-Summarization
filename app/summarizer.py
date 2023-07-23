import evaluate
import json
import networkx as nx
import nltk
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class GenSummarizer():
    '''
    A class for generating abstractive summaries using a pre-trained sequence-to-sequence model.

    Methods:
        __init__(self): Initializes the GenSummarizer class and downloads necessary NLTK data.
        get_tokenizer(self, name): Gets a pre-trained tokenizer for the specified model.
        generate_summary(self, model_name, text): Generates an abstractive summary using a pre-trained model and tokenizer.
        split_sentences(self, summary): Splits the summary into sentences.

    Example:
        summarizer = GenSummarizer()
        tokenizer = summarizer.get_tokenizer("t5-small")
        summary = summarizer.generate_summary("t5-small", text)
        sentences = summarizer.split_sentences(summary)
    '''
    def __init__(self):
        '''
        Initializes the GenSummarizer class and downloads necessary NLTK data.
        '''
        nltk.download('popular')

    def get_tokenizer(self, name):
        '''
        Gets a pre-trained tokenizer for the specified model.

        Args:
            name (str): Name of the pre-trained model.

        Returns:
            tokenizer: The pre-trained tokenizer.
        '''
        tokenizer = AutoTokenizer.from_pretrained(name)
        return tokenizer
    
    def generate_summary(self, model_name, text):
        '''
        Generates an abstractive summary using a pre-trained model and tokenizer.

        Args:
            model_name (str): Name of the pre-trained model.
            text (str): Input text for summarization.

        Returns:
            str: The generated summary.
        '''
        tokenizer = self.get_tokenizer(f'models/{model_name}')
        model = AutoModelForSeq2SeqLM.from_pretrained(f"models/{model_name}")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding= "max_length").input_ids
        outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    def split_sentences(self, summary):
        '''
        Splits the summary into sentences.

        Args:
            summary (str): The summary to split.

        Returns:
            list: A list of sentences.
        '''
        article_sentences = sent_tokenize(summary)
        return article_sentences
    
class ExtSummarizer():
    '''
    A class for generating extractive summaries using the TextRank algorithm.

    Methods:
        __init__(self): Initializes the ExtSummarizer class and downloads necessary NLTK data.
        get_sentences(self, article): Splits the article into sentences.
        preprocess(self, extracted_sentences): Preprocesses the extracted sentences for feature extraction.
        vectorize(self, processed_sents, vectorizer_type='count'): Vectorizes the processed sentences.
        generate_adjacency_matrix(self, feature_vecs): Generates the adjacency matrix.
        summarize(self, sentences, adjacency_matrix, top_n): Summarizes the article.
    '''
    def __init__(self):
        '''
        Initializes the ExtSummarizer class and downloads necessary NLTK data.
        '''
        nltk.download('popular')

    def get_sentences(self, article):
        '''
        Splits the article into sentences.

        Args:
            article (str): The article to split.

        Returns:
            list: A list of sentences.
        '''
        article_sentences = sent_tokenize(article)
        return article_sentences
    
    def preprocess(self, extracted_sentences):
        '''
        Preprocesses the extracted sentences for feature extraction.

        Args:
            extracted_sentences (list): A list of extracted sentences.

        Returns:
            list: A list of preprocessed sentences.
        '''
        sents_processed = []
        for sentence in extracted_sentences:
            s_reduced = sentence[0].replace("[^a-zA-Z0-9_]", '')
            s_reduced = [word.lower() for word in s_reduced.split(' ') if word.lower() not in stopwords.words('english')]
            sents_processed.append(' '.join(word for word in s_reduced))

        return sents_processed

    def vectorize(self, processed_sents, vectorizer_type='count'):
        '''
        Vectorizes the processed sentences.

        Args:
            processed_sents (list): A list of preprocessed sentences.
            vectorizer_type (str): The type of vectorizer to use ('count' or 'tfidf').

        Returns:
            list: A list of feature vectors.
        '''
        if vectorizer_type == 'count':
            # Get vocabulary for entire document
            processed_sents = [sent.split(' ') for sent in processed_sents]
            all_words = list(set([word for s in processed_sents for word in s]))

            # Create feature vector for each sentence
            feature_vecs = []
            for sentence in processed_sents:
                feature_vec = [0] * len(all_words)
                for word in sentence:
                    feature_vec[all_words.index(word)] += 1
                feature_vecs.append(feature_vec)
        else:
            vectorizer = TfidfVectorizer()
            feature_vecs = vectorizer.fit_transform(processed_sents)
            feature_vecs = feature_vecs.todense().tolist()
            
        return feature_vecs
    
    def generate_adjacency_matrix(self, feature_vecs):
        '''
        Generates the adjacency matrix.

        Args:
            feature_vecs (list): A list of feature vectors.

        Returns:
            numpy.ndarray: The adjacency matrix.
        '''
        # Create empty adjacency matrix
        adjacency_matrix = np.zeros((len(feature_vecs), len(feature_vecs)))
    
        # Populate the adjacency matrix using the similarity of all pairs of sentences
        for i in range(len(feature_vecs)):
            for j in range(len(feature_vecs)):
                if i == j: #ignore if both are the same sentence
                    continue 
                adjacency_matrix[i][j] = 1 - cosine_distance(feature_vecs[1], feature_vecs[j])
        
        return adjacency_matrix
    
    def summarize(self, sentences, adjacency_matrix, top_n):
        '''
        Summarizes the article.

        Args:
            sentences (list): A list of sentences in the article.
            adjacency_matrix (numpy.ndarray): The adjacency matrix.
            top_n (int): The number of sentences to include in the summary.

        Returns:
            str: The summarized article.
        '''

        # Create the graph representing the document
        document_graph = nx.from_numpy_array(adjacency_matrix)

        # Apply PageRank algorithm to get centrality scores for each node/sentence
        scores = nx.pagerank(document_graph)
        scores_list = list(scores.values())

        # Sort and pick top sentences
        ranking_idx = np.argsort(scores_list)[::-1]
        ranked_sentences = [sentences[i] for i in ranking_idx]   

        summary = []
        for i in range(top_n):
            summary.append(ranked_sentences[i])

        summary = " ".join(summary)

        return summary