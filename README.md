# tf-idf
# LeetCode Query Search

LeetCode Query Search is a web application that allows users to search for relevant LeetCode questions based on their queries. The application uses a TF-IDF model trained on a dataset of LeetCode questions to match the user's query with the most relevant questions.

## Features

- Scrapes LeetCode question data using Selenium and Beautiful Soup
- Trains a TF-IDF model on the question dataset
- Performs NLP processing to filter relevant words based on user queries
- Matches the top 20 results from the dataset based on the filtered words
- Provides a web interface for users to enter their queries and view the results
- Deployed on the Render platform for free hosting

## Technologies Used

- Python
- Selenium
- Beautiful Soup
- Flask
- nltk(for preprocessing)
- HTML/CSS (for the web interface)

Repository: https://github.com/machine2609/tf-idf

Hosted website: https://tf-idf-session-oeal.onrender.com/

//The loading of website takes sometime. Thanks for your patience.
## Steps to Build the App Locally
- Getting started
Clone this repo: https://github.com/machine2609/tf-idf/tree/master
1) Change to the repo directory: cd tf-idf
2) If you want to use virtual environment: conda create --name
3) activate the environment : conda activate --name
4) To deactivate the environment : conda deactivate
5) Install dependencies with pip or conda: pip install -r requirements.txt or conda install -r requirements.txt
6) Make sure to activate the environment. Then open the command line and run the app: python app.py
