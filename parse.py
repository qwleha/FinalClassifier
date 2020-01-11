import os
import sys

import requests
import pickle

import pandas as pd

from time import sleep

from selenium import webdriver

from bs4 import BeautifulSoup
import re


headings = ['/krasota-zdorovie/encyclopedia-zdorovia/', '/news/', '/moda-shopping/trendi/', '/stil-zhizny/intervyu-zvezd/', '/stil-zhizny/kop/']

basic_url = 'https://www.wday.ru'

dataset = {
    'здоровье': None,
    'новости': None,
    'тренды': None,
    'интервью': None,
    'лайфхаки': None,
}

data = pd.DataFrame(columns=['href', 'text', 'target'])
driver = webdriver.Chrome('./chromedriver')
for heading, type in zip(headings, dataset.keys()):
    driver.get(basic_url + heading)
    
    for _ in range(11):
        driver.find_element_by_class_name('subrubricAnnouncements__btn').click()
        sleep(1)
        
    soup = BeautifulSoup(driver.page_source)
    hrefs = list()
    for element in soup.findAll('div', {'class': 'subrubricAnnouncements__item'}):
        hrefs.append(element.find('a')['href'])
        
    articles = {}
    for href in hrefs[:100]:
        url = basic_url + href
        raw = requests.get(url)
        soup = BeautifulSoup(raw.text)
        text = ''
        for p in soup.find('article').findAll('p'):
            text += p.text + '\n'
        articles[basic_url + href] = text.replace('\\xa0', ' ')
        
    dataset[type] = articles
    
    for section, articles in dataset.items():
        for href, text in articles.items():
            data = data.append({'href': href, 'text': text, 'target': section}, ignore_index=True)
    data = data.sample(frac=1).reset_index(drop=True)

data.to_csv('dataset.csv', index=False)
