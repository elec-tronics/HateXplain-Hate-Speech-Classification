# HateXplain-Hate-Speech-Classification
NLP-based project for detecting hate speech, offensive, and harmful content in tweets and social media posts

## Overview
This project focuses on automatic hate speech classification in social media posts, particularly tweets. With the massive rise in user-generated content on platforms like Twitter, detecting abusive and offensive language has become increasingly important, yet manually impossible due to scale. The dataset used combines posts from Twitter and Gab, labeled into four classes — Normal, Offensive, Hate Speech, and Ambiguous.

The project applies Natural Language Processing (NLP) techniques along with machine learning and deep learning models to classify text effectively. For feature extraction, multiple embeddings were explored, including TF-IDF (uni/bi/tri-grams), Gensim-based Word2Vec (Continuous Bag of Words with 200-dimension vectors), and GloVe embeddings. Vectors were padded or truncated to ensure uniform size, and the processed feature set was exported to CSV for model training. The classification task was performed using various ML models, and results were compared to evaluate which combination of embeddings and models achieved better performance for the hate speech classification task.

