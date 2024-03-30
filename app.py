import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib 
import string
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def clean_lem(sentence):
    text = sentence.lower()

    soup = BeautifulSoup(text, 'html.parser')

    text = soup.get_text()

    text = ' '.join([word for word in text.split() if not word.startswith('http')])

    text = ''.join([char for char in text if char not in string.punctuation + '’‘'])

    text = ''.join([i for i in text if not i.isdigit()])

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    text = ' '.join([word for word in word_tokens if word.lower() not in stop_words])

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in tokens])

    return text

# Load the saved models

random_forest_model = joblib.load('Random_Classifier.joblib')
logistic_model = joblib.load('log_reg.joblib')
MLPClassifier_model = joblib.load('MLP_classifier.joblib')



# Define function for sentiment analysis
def predict_sentiment(text, model):
    if not text:
        return "Please enter some text for sentiment analysis."
    cleaned_lem = clean_lem(text)
    prediction = model.predict([cleaned_lem])
    return prediction[0]

# Define the Streamlit app
def main():
    st.title('Sentiment Analysis')

    # Select classifier
    classifier = st.sidebar.selectbox('Select Model', ['Logistic Regression', 'Random Forest','MLPClassifier'])

    # Input text
    text_input = st.text_area('Enter text for sentiment analysis')

    # Perform sentiment analysis
    if st.button('Analyze'):
        if text_input:
            if classifier == 'Logistic Regression':
                prediction = predict_sentiment(text_input, logistic_model)
                
            elif classifier == 'Random Forest':
                prediction = predict_sentiment(text_input, random_forest_model)
     
            elif classifier== 'MLPClassifier':
                prediction = predict_sentiment(text_input, MLPClassifier_model)

            st.write('Sentiment:', prediction)

    # Option to upload CSV file
    st.sidebar.header('Upload CSV file')
    uploaded_file = st.sidebar.file_uploader('Choose a CSV file', type='csv')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

        # Perform sentiment analysis on the CSV file
        if st.button('Analyze CSV'):
            if 'text' in df.columns:
                if classifier == 'Logistic Regression':
                    predictions = [predict_sentiment(text, logistic_model) for text in df['text']]
                elif classifier == 'Random Forest':
                    predictions = [predict_sentiment(text, random_forest_model) for text in df['text']]
                elif classifier == 'MLPClassifier':
                    predictions = [predict_sentiment(text, MLPClassifier_model) for text in df['text']]

                df['sentiment'] = predictions
                st.dataframe(df)

                # Draw bar chart for sentiment value counts
                st.subheader('Sentiment Distribution')
                sns.set(style="whitegrid")
                plt.figure(figsize=(6,4))
                sns.countplot(data=df, x='sentiment')
                plt.xlabel('Sentiment')
                plt.ylabel('Count')
                plt.title('Sentiment Distribution')
                st.pyplot(plt)
            else:
                st.warning('CSV file must contain a column named "text" for sentiment analysis.')

if __name__ == "__main__":
    st.set_page_config(
        page_title="Sentiment Analysis App",
        layout="centered",
        initial_sidebar_state="auto",
    )
    main()