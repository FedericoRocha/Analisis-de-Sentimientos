import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

st.title("Sentiment Analysis with NLTK")

st.header("Enter your text:")
user_input = st.text_area("Write the text to analyze:")

def analyze_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores

if user_input:
    sentiment_scores = analyze_sentiment(user_input)
    st.write("Sentiment Analysis Result:", sentiment_scores)

    
    categories = ['Negative', 'Neutral', 'Positive']
    scores = [sentiment_scores['neg'], sentiment_scores['neu'], sentiment_scores['pos']]

    fig = go.Figure(data=[go.Bar(x=categories, y=scores, text=scores, textposition='auto')])
    fig.update_layout(title="Sentiment Distribution",
                      xaxis_title="Categories",
                      yaxis_title="Score")
    st.plotly_chart(fig)

st.subheader("Sentiment Distribution Over Time (Example):")

import pandas as pd
import numpy as np
import random

dates = pd.date_range('2023-01-01', periods=10, freq='D')

random_sentiments = [sia.polarity_scores(" ".join([random.choice(["good", "bad", "happy", "sad", "neutral"]) for _ in range(5)])) for _ in range(10)]

neg_sentiments = [sentiment['neg'] for sentiment in random_sentiments]
pos_sentiments = [sentiment['pos'] for sentiment in random_sentiments]

fig_time = go.Figure()

fig_time.add_trace(go.Scatter(x=dates, y=neg_sentiments, mode='lines+markers', name='Negative'))
fig_time.add_trace(go.Scatter(x=dates, y=pos_sentiments, mode='lines+markers', name='Positive'))

fig_time.update_layout(title="Sentiment Analysis Over Time",
                      xaxis_title="Date",
                      yaxis_title="Sentiment Score")

st.plotly_chart(fig_time)
