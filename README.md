

## Project Report: Sentiment Analysis of Customer Reviews

1. Introduction

Customer reviews play a crucial role in understanding the perception of a product or service. Sentiment analysis of these reviews can provide valuable insights into customer satisfaction and identify areas for improvement. In this project, we aim to classify customer reviews for the "YONEX MAVIS 350 Nylon Shuttle" product from Flipkart as positive or negative. Additionally, we seek to understand the pain points of customers who write negative reviews.

2. Dataset

We obtained a dataset comprising 8,518 reviews from Flipkart. Each review includes features such as Reviewer Name, Rating, Review Title, Review Text, Place of Review, Date of Review, Up Votes, and Down Votes.

3. Data Preprocessing

We performed the following steps for data preprocessing:

Text Cleaning: Removed special characters, punctuation, and stopwords from the review text.
Text Normalization: Conducted lemmatization to reduce words to their base forms.
Numerical Feature Extraction: Employed Bag-of-Words (BoW), Term Frequency-Inverse Document Frequency (TF-IDF), Word2Vec (W2V), and BERT models for feature extraction.
4. Modeling Approach

Our modeling approach consisted of the following steps:

Model Selection: Trained and evaluated various machine learning and deep learning models using the embedded text data.
Evaluation Metric: Utilized the F1-Score as the evaluation metric to assess the performance of the models in classifying sentiment.
5. Model Deployment

For model deployment, we followed these steps:

Flask or Streamlit App Development: Developed a Flask web application that takes user input in the form of a review and generates the sentiment (positive or negative) of the review.
Model Integration: Integrated the trained sentiment classification model into the Flask app for real-time inference.
Deployment: Deployed the Flask app on an AWS EC2 instance to make it accessible over the internet.
6. Workflow

Our workflow involved the following stages:

Data Loading and Analysis: Gained insights into product features contributing to customer satisfaction or dissatisfaction.
Data Cleaning: Preprocessed the review text by removing noise and normalizing the text.
Text Embedding: Experimented with different text embedding techniques to represent the review text as numerical vectors.
Model Training: Trained machine learning and deep learning models on the embedded text data to classify sentiment.
Model Evaluation: Evaluated the performance of the trained models using the F1-Score metric.
Flask App Development: Developed a Flask web application for sentiment analysis of user-provided reviews.
Model Deployment: Deployed the trained sentiment classification model along with the Flask app on an AWS EC2 instance.
Testing and Monitoring: Tested the deployed application and monitored its performance for any issues or errors.
