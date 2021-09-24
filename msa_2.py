import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import itertools
url = '''https://www.gutenberg.org/ebooks/search/?sort_order=downloads'''
respond = requests.get(url)
book_soup = BeautifulSoup(respond.text, 'html.parser')
# Filter the a-tags to get just the titles
book_tags = book_soup.find_all('a', attrs={'class': "link"})
book_tags = [tag.attrs['href'] for tag in book_tags 
              if tag.attrs['href'].startswith('/ebook') & (tag.attrs['href'][-1].isdigit()==True)]
# Remove duplicate links
book_tags = list(dict.fromkeys(book_tags))
print("In total we have " + str(len(book_tags)) + " book titles") 
print(book_tags[:10])

base_url = "https://www.gutenberg.org"
# Get book links
book_links = [base_url + tag  for tag in book_tags]
print(book_links[:10])

# Create a helper function to get review links
def getBook(soup):
    # Get all the review tags
    book_list = soup.find_all('a', attrs={'class':'link'})
    # Get the first review tag
    format_tag = book_list[0]
    # Return the none review link
    format_link = "https://www.gutenberg.org" + format_tag['href']
    return format_link

# Get a list of soup objects. This takes a while
book_form_soups = [BeautifulSoup(requests.get(link).text, 'html.parser') for link in book_links]
# Get all 100 movie review links
book_format_list = [getBook(book_form_soup) for book_form_soup in book_form_soups]

print("There are a total of " + str(len(book_format_list)) + " individual movie reviews")
print(book_format_list[:10])

# Create lists for dataframe and csv later
book_texts = []
book_titles = []
regex = re.compile('chapter.*')

# Loop through the movie reviews
for url in book_format_list:
    # Get the review page
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    # Find div tags with class text show-more__control, then get its text
    if soup.find_all('div', attrs={'class': regex}):
      chapter_tag = soup.find('div', attrs={'class': 'chapter'}).getText()
    else:
      chapter_tag = soup.find_all('p')[5].getText()
    # Add the book text in the book list
    book_texts += [chapter_tag]
    # Find the h1 tag and get the title
    title_tag = soup.find('h1').getText()
    # Add the title in the title list
    book_titles += [title_tag]

# Construct a dataframe
df = pd.DataFrame({'Book': book_titles, 'book_download_link':book_format_list,'book_chapter': book_texts})
# Put into .csv file
with open("Book_download.csv", mode='w', newline='\n') as f:
    df.to_csv(f, sep=",", line_terminator='\n', encoding='utf-8')
#df.to_csv('Book_download.csv',encoding='utf-8', index=False)
data=pd.read_csv('Book_download.csv')
for line in data:
  line = re.sub(r"\\n\r", "", line)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(lowercase=True,token_pattern=r'(?u)\b[A-Za-z]+\b',stop_words='english',max_features=2000,strip_accents='unicode')
X = vectorizer.fit_transform(data['book_chapter'].values)
print(vectorizer.get_feature_names())

