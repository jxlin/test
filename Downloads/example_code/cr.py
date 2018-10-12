# -*-coding:UTF-8 -*-
import re
import urllib
def get_html(url):
    page = urllib.urlopen(url)
    html = page.read()
    return html
 
def get_images(html):
    reg = r'src="(.+?\.jpg)" pic_ext'
    imgre = re.compile(reg)
    img_list = imgre.findall(html)
    x = 0
    for imgurl in img_list:
        urllib.urlretrieve(imgurl,'/Users/jensenlin/tmp/%s.jpg' % x)       
        x = x+1
 
if __name__ == '__main__':   
    html = get_html("http://tieba.baidu.com/p/2460150866")
    get_images(html)
