import evaluate
import json
import networkx as nx
import nltk
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class ExtractiveModel():
    '''
    A class representing an Extractive Text Summarization Model using cosine similarity and PageRank.

    Attributes:
        test_set (Dataset): The test dataset used for evaluation.

    Methods:
        __init__(self, datasets): Initializes the ExtractiveModel with the test dataset.
        get_sentences(self, article): Tokenizes the input article into sentences.
        preprocess(self, extracted_sentences): Preprocesses the extracted sentences.
        vectorize(self, processed_sents, vectorizer_type='count'): Vectorizes the preprocessed sentences.
        generate_adjacency_matrix(self, feature_vecs): Generates the adjacency matrix based on the similarity of sentences.
        summarize(self, sentences, adjacency_matrix, top_n): Generates the summary by applying PageRank algorithm.
        test(self, save_name): Tests the model and saves evaluation results.
    '''
    
    def __init__(self, datasets):
        '''
        Initializes the ExtractiveModel with the test dataset.

        Args:
            datasets: HuggingFace dataset containing 'test' dataset.
        '''
        nltk.download('popular')
        self.test_set = datasets['test']

    def get_sentences(self, article):
        '''
        Tokenizes the input article into sentences.

        Args:
            article (str): The input article to be tokenized.

        Returns:
            list: A list of sentences extracted from the article.
        '''
        article_sentences = sent_tokenize(article)
        return article_sentences
    
    def preprocess(self, extracted_sentences):
        '''
        Preprocesses the extracted sentences.

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
        Vectorizes the preprocessed sentences.

        Args:
            processed_sents (list): A list of preprocessed sentences.
            vectorizer_type (str): The type of vectorizer to use ('count' or 'tfidf').

        Returns:
            list: A list of feature vectors for each sentence.
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
        Generates the adjacency matrix based on the similarity of sentences.

        Args:
            feature_vecs: feature vectors for each sentence.

        Returns:
            adjacency_matrix: The adjacency matrix representing sentence similarity.
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
        Generates the summary by applying PageRank algorithm.

        Args:
            sentences (list): A list of sentences in the article.
            adjacency_matrix (numpy.ndarray): The adjacency matrix representing sentence similarity.
            top_n (int): The number of top sentences to include in the summary.

        Returns:
            str: The generated summary.
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
    
    def test(self, save_name):
        '''
        Tests the model and saves evaluation results.

        Args:
            save_name (str): Name of the evaluation file to save.
        '''

        rouge = evaluate.load("rouge")
        refs = []
        summaries = []

        for d in self.test_set:
            ext_sents = self.get_sentences(d['text'])
            processed_article = self.preprocess(ext_sents)  
            fvs = self.vectorize(processed_sents = processed_article)
            adj_mat = self.generate_adjacency_matrix(feature_vecs = fvs)
            pred_summary = self.summarize(ext_sents, adj_mat, top_n = 5)
            refs.append(d['summary'])
            summaries.append(pred_summary)

        rouge_scores = rouge.compute(predictions=summaries, references=refs)
        
        with open(f'evals/{save_name}.json', 'w') as file:
            json.dump(rouge_scores, file)