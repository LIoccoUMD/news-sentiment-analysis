import nltk
# nltk.download('punkt_tab')
# import spaCy
import numpy
import requests
from bs4 import BeautifulSoup
import time
import lxml # BeautifulSoup uses this to parse xml
from transformers import pipeline
import torch
from datasets import Dataset # Models perform better on datasets
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mysql.connector as mysql
from mysql.connector import errorcode
from tabulate import tabulate
import os
from dotenv import load_dotenv

#----------#
#   Vars   #
#----------#

# Database Credential Retrieval
load_dotenv()
db_host = os.getenv("DB_HOST")
db_database = os.getenv("DB_DATABASE")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
try: 
    db_connection = mysql.connect(host=db_host,database=db_database,password=db_password,user=db_user)
    print(db_connection.get_server_info())
    cursor = db_connection.cursor()
    cursor.execute("SELECT * FROM articles")
    print(cursor.fetchall())
    db_connection.close()
except mysql.Error as e :
    print(f"Failed to connect to database: {e}")
    exit(1)

newsStations = {
    "CNN":"https://www.cnn.com/sitemap/news.xml",
    "FOX":"https://www.foxnews.com/sitemap.xml?type=news",
    "BBC":"https://www.bbc.com/sitemaps/https-sitemap-com-news-2.xml", # BBC has 3 news urls (news-1, news-2, news-3).
    "REUTERS":"https://www.reuters.com/arc/outboundfeeds/news-sitemap/?outputType=xml",
    "MSNBC":"https://www.msnbc.com/sitemap/msnbc/sitemap-news",
    "SkyNews":"https://news.sky.com/sitemap/sitemap-news.xml"
}

def scrapeTitlesXML(url):
    response = requests.get(url)
    # Store the xml 
    content = response.text
    # Soupify!
    soup = BeautifulSoup(content, "xml")
    titles = soup.find_all("news:title") # 297 article titles total
    return titles


# Load the sentiment analysis pipeline
def titleSentimentAnalysis(titles):
    headlines = [title.text for title in titles]
    dataset = Dataset.from_dict({"articleTitles": headlines})
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    if torch.cuda.is_available(): # TO-DO: Stop this from printing
        classifier.model.to("cuda")
    
    results = classifier(dataset["articleTitles"], batch_size=32)
    return results # These are the scores of whether or not the title is positive or negative

def plotSentiment(stations, pos_counts, neg_counts):
    plt.figure(figsize=(10,6))
    plt.bar(stations.keys(), pos_counts, label="Positive", color="green")
    plt.bar(stations.keys(), neg_counts, label="Negative", color="red")

    plt.xlabel("News Stations")
    plt.ylabel("Number of Article Titles")
    plt.title("Sentiment Analysis of News Titles by Station")
    plt.legend()
    # plt.xticks(rotation=45)   
    plt.show()

# Prints every title along with its lable (POSTIIVE OR NEGATIVE) and it's score (-1 to 1)
def printAllTitles(titles, scores):
    for i, (title, result) in enumerate(zip(titles, scores)):
        print(f"Title {i+1}: {title.text}")
        print(f"  Label: {result['label']}")
        print(f"  Score: {result['score']}\n")

# Prints the number of article titles with positive and negative sentiment analysis.
def printPosandNeg(scores, station):
    numPositive = sum(1 for result in scores if result["label"] == "POSITIVE")
    numNegative = sum(1 for result in scores if result["label"] == "NEGATIVE")
    print(f"{station} number of POSITIVE article titles: {numPositive}")
    print(f"{station} number of NEGATIVE article titles: {numNegative}\n")
    return numPositive, numNegative


def makeWordcloud(url):
    titles = scrapeTitlesXML(url) 
    text = " ".join(title.get_text() for title in titles)
    wordcloud = WordCloud(width=800, height=400,background_color="white",min_font_size=10,collocations=False).generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    return wordcloud


def main():
    # print(f"CUDA availability: {torch.cuda.is_available()}")
    # pos_counts = []
    # neg_counts = []
    # for station, url in newsStations.items():
    #     try:
    #         titles = scrapeTitlesXML(url)
    #         scores = titleSentimentAnalysis(titles)
    #         # printAllTitles(titles, scores)
    #         # makeWordcloud(url)
    #         numPos, numNeg = printPosandNeg(scores,station)
    #         pos_counts.append(numPos)
    #         neg_counts.append(numNeg)
    #     except Exception as e:
    #         print(f"Error processing {station}: {e}")
    # plotSentiment(newsStations, pos_counts, neg_counts)



    if __name__ == "__main__":
        main()