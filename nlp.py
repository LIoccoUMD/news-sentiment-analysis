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


#----------#
#   Vars   #
#----------#
newsStations = {
    "CNN":"https://www.cnn.com/sitemap/news.xml",
    "FOX":"https://www.foxnews.com/sitemap.xml?type=news",
    "BBC":"https://www.bbc.com/sitemaps/https-sitemap-com-news-2.xml", # BBC has 3 news urls (news-1, news-2, news-3).
    "REUTERS":"https://www.reuters.com/arc/outboundfeeds/news-sitemap/?outputType=xml",
    "MSNBC":"https://www.msnbc.com/sitemap/msnbc/sitemap-news"
}



# News stations. Should probably make this a list or maybe even dictionary?
     # newsStations = {BBC1:"...news-1", BBC2:"...news2" BBC3:"...news3"}


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

    return results

def visualize(stations, pos_counts, neg_counts):
    return

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

def main():
    print(f"CUDA availability: {torch.cuda.is_available()}")
    
    for station, url in newsStations.items():
        titles = scrapeTitlesXML(url)
        scores = titleSentimentAnalysis(titles)
        printPosandNeg(scores,station)

main()