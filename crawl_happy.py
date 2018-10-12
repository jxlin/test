#!/usr/bin python
#--*-- coding:utf-8 --*--
import os
import urllib
import re
import time
import urllib2    
import HTMLParser
 
 
#获取页面内容    
def gethtml(url):        
    print u'start crawl %s ...' % url    
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.2; rv:16.0) Gecko/20100101 Firefox/16.0'}    
    req = urllib2.Request(url=url,headers=headers)    
    try:    
        html = urllib2.urlopen(req).read().decode('utf-8')    
        html=HTMLParser.HTMLParser().unescape(html)#处理网页内容， 可以将一些html类型的符号如" 转换回双引号      
    except urllib2.HTTPError,e:    
        print u"连接失败，错误原因：%s " % e.code    
        return None    
    except urllib2.URLError,e:    
        if hasattr(e,'reason'):    
            print u"连接失败，错误原因:%s " % e.reason    
            return None    
    return html    
 
def getImageList(html):
    #reg = "http:*?\.jpg"
    reg = 'http[^"}]*?(?:\.jpg|\.png|\.jpeg)'#匹配图片url的正则表达式
    imgre = re.compile(reg)
    imgList = re.findall(imgre,html)
    return imgList
#打印所有的图片的地址并存贮到本地 
def printImageList(imgList):
    with open("webImage/url.txt","wb+") as f:
       for i in imgList:
            print i
            f.write(i+"\r\n")
#下载存贮图片到本地
def download(imgList, page):
    x = 1
    for imgurl in imgList:
        print 'Download '+imgurl
        urllib.urlretrieve(imgurl,'./webImage/%s_%s.jpg'%(page,x))
        x+=1
    print 'Download file '+ str(x)+ ' fiel\'s end'
 
 
def downImageNum(pagenum):
    page = 1
    pageNumber = pagenum
    while(page <= pageNumber):
        html = getHtml(url)#获得url指向的html内容
        imageList = getImageList(html)
        printImageList(imageList)#打印所有的图片的地址
        download(imageList,page)#下载所有的图片
        page = page+1
 
if __name__ == '__main__':
    print '''  
            *****************************************   
            **   Welcome to python of Image        **   
            **      Modify on 2017-05-09           **   
            **      @author: Jimy _Fengqi          **   
            *****************************************  
    '''   
    os.system('mkdir webImage')#创建文件存贮目录
    url = raw_input("enter the web page\n URL:")
	if not url:
		print 'the url in None , please try again'
		break
    downImageNum(1)
    time.sleep(10)
