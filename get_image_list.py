#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import sys
from google_images_download import google_images_download

list1=[]
input=sys.argv[1]
response = google_images_download.googleimagesdownload()
#class instantiation

with open(input, "rb") as fp:
    name = fp.readline().decode("UTF-8").strip()
    cnt = 1

    while name:
        name = u''.join(name).encode('utf-8')
        msg = "------ Downloading images of person #" + str(cnt) + ' ------'
        print msg
        arg_medium = {"keywords":name,"limit":60,"size":"medium","print_urls":True}   #creating list of arguments - medium
        paths = response.download(arg_medium)   #passing the arguments to the function
        #print(paths)

        arg_large = {"keywords":name,"limit":30,"size":"large","print_urls":True}   #creating list of arguments - large
       #printing absolute paths of the downloaded images
        paths = response.download(arg_large)   #passing the arguments to the function
        #print(paths)
        name = fp.readline().decode("UTF-8").strip()
        cnt+=1
