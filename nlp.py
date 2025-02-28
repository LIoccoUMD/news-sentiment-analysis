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


cnn = "https://www.cnn.com/sitemap/news.xml"
fox = "https://www.foxnews.com/sitemap.xml?type=news"
bbc = "https://www.bbc.com/sitemaps/https-sitemap-com-news-2.xml" # BBC has 3 news urls (news-1, news-2, news-3).



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
    if torch.cuda.is_available(): # Stop this from printing
        classifier.model.to("cuda")
    
    results = classifier(dataset["articleTitles"], batch_size=32)

    return results

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
    print(f"{station} Count of POSITIVE scores: {numPositive}")
    print(f"{station} Count of NEGATIVE scores: {numNegative}")

def main():
    print(f"CUDA availability: {torch.cuda.is_available()}")
    # CNN news
    cnnTitles = scrapeTitlesXML(cnn)
    cnnScores = titleSentimentAnalysis(cnnTitles)
    # Fox news
    foxTitles = scrapeTitlesXML(fox)
    foxScores = titleSentimentAnalysis(foxTitles)

    # BBC news
    bbcTitles = scrapeTitlesXML(bbc)
    bbcScores = titleSentimentAnalysis(bbcTitles)


    printPosandNeg(cnnScores, "CNN")
    printPosandNeg(foxScores, "FOX")
    printPosandNeg(bbcScores, "BBC")

main()