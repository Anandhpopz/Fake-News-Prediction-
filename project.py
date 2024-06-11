import streamlit as st
import pickle
import spacy

with open(r"C:\Users\91735\Downloads\svm_model.pkl", 'rb') as f:
    sv = pickle.load(f)
    
nlp = spacy.load('en_core_web_sm')

def predict_news(input_text):
    x_input = nlp(input_text).vector.reshape(1, -1)
    prediction = sv.predict(x_input)
    return prediction[0]

st.title("News Classifier")
news_input = st.text_area("Enter the news:")
if st.button("Classify"):
    prediction = predict_news(news_input)
    if prediction == 1:
        st.write("Fake News")
    else:
        st.write("Real News")
