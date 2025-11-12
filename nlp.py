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
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mysql.connector as mysql
from mysql.connector import errorcode
from tabulate import tabulate
import os
from dotenv import load_dotenv
from datetime import date

#----------#
#   Vars   #
#----------#

# Database Credential Retrieval
def connect_DB():
    load_dotenv()
    db_host = os.getenv("DB_HOST")
    db_database = os.getenv("DB_DATABASE")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    try: 
        cnx = mysql.connect(host=db_host,database=db_database,password=db_password,user=db_user)
        print(f"Connected to MySQL ServeR: {cnx.get_server_info()}")
        return cnx
    except mysql.connector.Error as err :
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
        return None

# newsStations = {
#     "CNN":"https://www.cnn.com/sitemap/news.xml",
#     "FOX":"https://www.foxnews.com/sitemap.xml?type=news",
#     "BBC":"https://www.bbc.com/sitemaps/https-sitemap-com-news-2.xml", # BBC has 3 news urls (news-1, news-2, news-3).
#     "REUTERS":"https://www.reuters.com/arc/outboundfeeds/news-sitemap/?outputType=xml",
#     "MSNBC":"https://www.msnbc.com/sitemap/msnbc/sitemap-news",
#     "SkyNews":"https://news.sky.com/sitemap/sitemap-news.xml",
#     "CBS":"https://www.cbsnews.com/xml-sitemap/news.xml"
# }

"""
Returns a dictionary of all processed news stations where the key is the station and the value is the sitemap that gets scraped.
"""
def get_news_stations():
    return {
        "CNN": "https://www.cnn.com/sitemap/news.xml",
        "FOX": "https://www.foxnews.com/sitemap.xml?type=news",
        "BBC": "https://www.bbc.com/sitemaps/https-sitemap-com-news-2.xml",
        "REUTERS": "https://www.reuters.com/arc/outboundfeeds/news-sitemap/?outputType=xml",
        "MSNBC": "https://www.msnbc.com/sitemap/msnbc/sitemap-news",
        "SkyNews": "https://news.sky.com/sitemap/sitemap-news.xml",
        "CBS": "https://www.cbsnews.com/xml-sitemap/news.xml"
    }

def scrapeTitlesXML(url):
    response = requests.get(url)
    # Store the xml 
    content = response.text
    # Soupify!
    soup = BeautifulSoup(content, "xml")
    titles = soup.find_all("news:title") # 297 article titles total
    return titles


"""
Loads the sentiment analysis pipeline. This function performs the sentiment analysis by processing
each word in the title of the article using a distilbert model and determining a POSTIVE or NEGATIVE
score for it. Returns a float between -1 and 1.
"""
def titleSentimentAnalysis(titles):
    headlines = [title.text for title in titles]
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    # print(torch.__version__)  # Should end +cu121
    # print(torch.cuda.is_available())  # True
    if torch.cuda.is_available(): # TO-DO: Stop this from printing
        classifier.model.to("cuda")
    
    results = classifier(headlines, batch_size=32)
    return results # These are the scores of whether or not the title is positive or negative

def insertDatabase():
    
    return

"""
Calculates a negativity score for each source and then calculates the percentage difference between
the specified two sources. The results argument should be taken from the titleSentimentAnalysis function.
"""
def score(results:dict,source1:str,source2:str):
    source1_neg = results[source1]['NEGATIVE']
    source1_pos = results[source1]['POSITIVE']
    source2_neg = results[source2]['NEGATIVE']
    source2_pos = results[source2]['POSITIVE']

    ns_source1 = source1_neg / (source1_neg + source1_pos)
    ns_source2 = source2_neg / (source2_neg + source2_pos)

    percent_diff = ((ns_source1 - ns_source2) / ns_source2) * 100

    comparative_statement = f"{source1} News sentiment is {abs(percent_diff):.2f}% "

    if percent_diff > 0:
        comparative_statement += f"more negative than {source2}."
    else:
        comparative_statement += f"less negative than {source2}."

    print(comparative_statement)
    return percent_diff



"""
Prints every title along with its label, POSITIVE or NEGATIVE, and its score which is between -1 and 1.
"""
def printAllTitles(titles, scores, _numtitles):
    for i, (title, result) in enumerate(zip(titles, scores)):
        if i >= _numtitles:
            break
        print(f"Title {i+1}: {title.text}")
        print(f"  Label: {result['label']}")
        print(f"  Score: {result['score']}\n")

"""
Prints the number of article titles with positive and negative sentiment analysis.
Returns a tuple of the number of articles perceived as POSITIVE and NEGATIVE for station.
"""
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

def compareSentiment(stations, pos_counts, neg_counts):
    fig, ax = plt.subplots()
    bottom = numpy.zeros(len(stations))    
    labels = list(stations.keys())
    p = ax.bar(labels,pos_counts,neg_counts, 0.5, label="Boolean",bottom=bottom)
    print(labels)
    plt.show()
    
    pass

def main():
    # cnx = connect_DB()
    # if cnx is None:
        # return
    try:
        print(f"CUDA availability: {torch.cuda.is_available()}")
        all_station_results = {}
        pos_counts = []
        neg_counts = []
        stations = get_news_stations()
        for station, url in stations.items():
            try:
                titles = scrapeTitlesXML(url)
                scores = titleSentimentAnalysis(titles)
                # printAllTitles(titles, scores,100)
                # makeWordcloud(url)
                numPos, numNeg = printPosandNeg(scores,station)

                all_station_results[station] = {
                    "POSITIVE": numPos,
                    "NEGATIVE": numNeg
                }
                pos_counts.append(numPos)
                neg_counts.append(numNeg)
            except Exception as e:
                print(f"Error processing {station}: {e}")
                print("nice!")
        score(all_station_results,"BBC","CBS")
    except:
        print("ok")


if __name__ == "__main__": main()