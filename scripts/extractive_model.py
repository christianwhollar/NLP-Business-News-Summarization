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
    def __init__(self, datasets):
        nltk.download('popular')
        self.test_set = datasets['test']

    def get_sentences(self, article):
        '''
        
        '''
        article_sentences = sent_tokenize(article)
        return article_sentences
    
    def preprocess(self, extracted_sentences):
        '''
        
        '''
        sents_processed = []
        for sentence in extracted_sentences:
            s_reduced = sentence[0].replace("[^a-zA-Z0-9_]", '')
            s_reduced = [word.lower() for word in s_reduced.split(' ') if word.lower() not in stopwords.words('english')]
            sents_processed.append(' '.join(word for word in s_reduced))

        return sents_processed

    def vectorize(self, processed_sents, vectorizer_type='count'):
        '''
        
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