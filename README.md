# HateXplain-Hate-Speech-Classification
NLP-based project for detecting hate speech, offensive, and harmful content in tweets and social media posts

## Overview
This project focuses on automatic hate speech classification in social media posts, particularly tweets. With the massive rise in user-generated content on platforms like Twitter, detecting abusive and offensive language has become increasingly important, yet manually impossible due to scale. The dataset used combines posts from Twitter and Gab, labeled into four classes — Normal, Offensive, Hate Speech, and Ambiguous.

The project applies Natural Language Processing (NLP) techniques along with machine learning and deep learning models to classify text effectively. For feature extraction, multiple embeddings were explored, including TF-IDF (uni/bi/tri-grams), Gensim-based Word2Vec (Continuous Bag of Words with 200-dimension vectors), and GloVe embeddings. Vectors were padded or truncated to ensure uniform size, and the processed feature set was exported to CSV for model training. The classification task was performed using various ML models, and results were compared to evaluate which combination of embeddings and models achieved better performance for the hate speech classification task.

## Dataset and Preparation
Dataset Link: https://github.com/hate-alert/HateXplain/tree/master

The dataset used in this project is sourced from the HateXplain repository. It is available as a single JSON file of 12 MB containing 20,148 records collected from Twitter and Gab. Each record includes a unique post ID, the tokenized text of the post, annotations from three human annotators, and additional metadata. The annotators categorized each post into one of three classes: normal, offensive, or hate speech. This dataset provides a reliable foundation for experimenting with NLP models and evaluating different feature extraction and classification methods.

To prepare the dataset for model input, the original JSON file was loaded into a Pandas DataFrame, and key fields were selected for analysis. Each post was assigned a final label through majority voting across the three annotators. In cases where all three labels differed, an additional ambiguous class was introduced. The resulting distribution included approximately 8,000 normal, 6,000 offensive, 6,000 hate speech, and 1,000 ambiguous posts. These class proportions were preserved to maintain data integrity, with no synthetic balancing applied. Finally, labels were converted into ordinal integers (0–3) to meet model input requirements, and the transformed dataset was stored in CSV format for further exploration and training.

## Tech Stack
Programming Language: Python

Frameworks & Libraries: Jupyter Notebook, scikit-learn, NumPy, Pandas, Matplotlib, Seaborn

Core Concepts: Supervised Machine Learning, Multiclass Classification, Feature Extraction

Machine Learning Algorithms: K-Nearest Neighbors (KNN), Support Vector Classifier (SVC), Decision Trees, Naive Bayes, Random Forest (RF), AdaBoost

## Approach
Exploratory Data Analysis (EDA): Data overview, label distribution, class imbalance check, annotation agreement analysis

Data Preprocessing: Tokenization, stopword removal, stemming, truncation/padding to 200 tokens, bi/tri-gram generation

Feature Extraction: TF-IDF (uni/bi/tri-grams), Word2Vec (200-dim CBOW via Gensim), GloVe embeddings (200-dim), LSTM feature maps

Modeling: Machine Learning (KNN, SVC, Decision Trees, Naive Bayes, Random Forest, AdaBoost) and Deep Learning (LSTM with GloVe embeddings)

Evaluation Metrics: Accuracy, Precision, Recall, F1 score, Macro F1 score

## Contributing
Contributions are welcome! Please fork the repo and submit a pull request.
