import requests
import csv
import time
from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0'}
url = "https://tas-creditsuisse.taleo.net/careersection/external_advsearch/jobdetail.ftl"
page = requests.get(url, headers=headers)       #get raw data


soup = BeautifulSoup(page.text, 'html.parser')  #make it soup

title = "View this job description"
all = soup.find_all(title)

p = soup.find("span", { "class" : "blockpanel" }) #within span, look for class = blockpanel


print p
