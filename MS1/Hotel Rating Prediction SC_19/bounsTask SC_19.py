
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


data = pd.read_csv("D:\\Collage\\Sem 6\\machine learning\\Project\\MS1\\hotel-regression-dataset.csv")


sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores['compound']

data['Positive_Review'] = data['Positive_Review'].apply(analyze_sentiment)
data['Negative_Review'] = data['Negative_Review'].apply(analyze_sentiment)

print(data['Negative_Review'])
print(data['Negative_Review'])

