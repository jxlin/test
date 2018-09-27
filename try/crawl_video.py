import requests
from bs4 import BeautifulSoup
import sys
import urllib

url = 'http://www.4s4s.cc/vodtypehtml/16-2.html'
strs = 'http://www.4s4s.cc'

wd_data = requests.get(url)
soup = BeautifulSoup(wd_data.text,'lxml')
titles = soup.select('body > div > div.index-area.clearfix > ul > li > a > span.lzbz > p.name')
imgs = soup.select('body > div > div.index-area.clearfix > ul > li > a > img')
links = soup.select('body > div > div.index-area.clearfix > ul > li > a')
#print(titles,imgs,links)
for title,img,link in zip(titles,imgs,links):
    data = {
        'title':title.get_text(),
        'img':img.get('src'),
        'link':strs+link.get('href')
    }
    print(data)

    x = 1
    print (data.get('link'))
    image_name = '%s.mp4' % x
    urllib.request.urlretrieve(data.get('link'),image_name)
    x+=1