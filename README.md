# News Sentiment Analyzer

This Python project scrapes article titles from news sitemaps, performs sentiment analysis on the headlines, and provides a breakdown of positive and negative sentiments. Using web scraping and natural language processing (NLP) techniques, it offers insights into the emotional tone of recent news coverage. Analysis is performed on current events.

## Features

- Scrapes article titles from XML sitemaps:
  - CNN: `https://www.cnn.com/sitemap/news.xml`
  - Fox News: `https://www.foxnews.com/sitemap.xml?type=news`
  - BBC: `https://www.bbc.com/sitemaps/https-sitemap-com-news-2.xml`
  - Reuters: `https://www.reuters.com/arc/outboundfeeds/news-sitemap/?outputType=xml`
- Performs sentiment analysis using the `distilbert-base-uncased-finetuned-sst-2-english` model from Hugging Face's Transformers library.
- Optimizes performance with batch processing and GPU support (CUDA) when available.
- Outputs the count of positive and negative headlines for each news source.

## Technologies Used

- **Python Libraries**:
  - `requests` for HTTP requests
  - `BeautifulSoup` for XML parsing
  - `transformers` for sentiment analysis
  - `torch` for GPU acceleration
  - `datasets` for efficient data handling
  - `nltk` for tokenization
  - `numpy` for numerical operations
- **NLP**: Pre-trained DistilBERT model for sentiment classification
- **Web Scraping**: XML parsing with BeautifulSoup and `lxml`

## How It Works

1. Fetches XML sitemap data using `requests`.
2. Extracts article titles with `BeautifulSoup`.
3. Converts titles into a `Dataset` object for batch processing.
4. Runs sentiment analysis to classify titles as "POSITIVE" or "NEGATIVE" with confidence scores.
5. Summarizes results by counting positive and negative sentiments for each news source.

## Installation

1. Clone the repository: ```git clone https://github.com/LIoccoUMD/news-sentiment-analyzer.git```
2. Navigate to the directory: ``` cd news-sentiment-analyzer ```
3. Install the required dependencies: ```pip install -r requirements.txt```

## Example Output
```
CUDA availability: True
Device set to use cuda:0
Device set to use cuda:0
Device set to use cuda:0
Device set to use cuda:0
CNN Count of POSITIVE scores: 91
CNN Count of NEGATIVE scores: 185
FOX Count of POSITIVE scores: 99
FOX Count of NEGATIVE scores: 223
BBC Count of POSITIVE scores: 389
BBC Count of NEGATIVE scores: 608
Reuters Count of POSITIVE scores: 11
Reuters Count of NEGATIVE scores: 39
```
