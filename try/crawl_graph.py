headers = {
         'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'
}

def req_tieba_img(url,page):
    html = requests.get(url,headers=headers)
    html.encoding = "utf-8"
    response = html.text
    reg1 = r'<div id="plist".+?<div class="page clearfix">'
    data1 = re.compile(reg1,re.S).findall(response)
    reg2 = r'<img width="220" height="220" data-img="1" src="//(.*?\.jpg)"'
    list_img = re.compile(reg2,re.S).findall(data1[0])
    x  = 1
    for imgurl in list_img:
        image_name = "D://ML/02/untitled2"+str(page)+str(x)+".jpg"
        imgurl = "http://"+imgurl
        print("正在写入第"+str(page)+"页第"+str(x)+"张图片")
        try:
            urllib.request.urlretrieve(imgurl,filename=image_name)
        except Exception as e:
            x += 1
        x+=1

for i in range(1,67):
    url = "https://list.jd.com/list.html?cat=9987,653,655&page="+str(i)
    req_tieba_img(url,i)