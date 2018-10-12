# -*-coding:UTF-8 -*-
import os
import re
import sys
import urllib
import argparse
def get_html(url):
    page = urllib.urlopen(url)
    html = page.read()
    return html
 
def get_images(html):
    ans = []
    reg = r'src="(.+?\.jpg)" pic_ext'
    imgre = re.compile(reg)
    img_list = imgre.findall(html)
    x = 0
    for imgurl in img_list:
        urllib.urlretrieve(imgurl,'/home/jingxian/example_code/tmp/%s.jpg' % x)
        x = x+1
        ans.append(imgurl)
    return ans

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("-l", "--url", required=True)

    args = parser.parse_args()

    html = get_html(args.url)
    ans = get_images(html)

    os.system("python classify_nsfw.py -m data/open_nsfw-weights.npy tmp")

    file = open('Results.txt', 'r')

    for f in file:
        s = f.split('\t')
        if int(s[0][4:-4]) < len(ans): print(ans[int(s[0][4:-4])] + '\tinappropriate score:\t' + s[-1])

if __name__ == '__main__':   
    main(sys.argv)
