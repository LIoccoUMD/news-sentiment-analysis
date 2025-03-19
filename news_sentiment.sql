-- Dedicated user for security
-- CREATE USER 'news_user'@'localhost' IDENTIFIED BY 'password';
-- GRANT ALL PRIVILEGES ON news_sentiment.* TO 'news_user'@'localhost';
-- FLUSH PRIVILEGES;


-- CREATE DATABASE news_sentiment;
-- USE news_sentiment;
-- CREATE TABLE articles (
-- id INT AUTO_INCREMENT PRIMARY KEY,
-- title VARCHAR(255),
-- source VARCHAR(50),
-- sentiment_label VARCHAR(10),
-- sentiment_score FLOAT,
-- scraped_date DATE
-- );
-- SHOW TABLES;

SELECT * FROM articles