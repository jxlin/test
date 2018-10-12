# -*- encoding:utf-8 -*-

import os
import re
import requests
from hashlib import md5
from urllib.parse import urlencode
from multiprocessing.pool import Pool
 
 
def get_page(offset):
	params = {
		'offset': offset,
		'format': 'json',
		'keyword': '街拍美女',
		'autoload': 'true',
		'count': '20',
		'cur_tab': '1',
	}
	url = 'https://www.toutiao.com/search_content/?' + urlencode(params)
	try:
		response = requests.get(url)
		if response.status_code == 200:
			return response.json()
	except requests.ConnectionErprror:
		return None
 
 
def get_images(json):
	if json and json.get('data'):
		for item in json.get('data'):
			if item.get('title'):
				title = item.get('title')
				title = re.sub(r'[? " / \\ < > * | :]', '', title)
			if item.get('image_list'):
				images = item.get('image_list')
				for image in images:
					yield {
						'image': 'http:' + image.get('url').replace('list', 'large'),
						'title': title
					}
 
 
def save_image(item):
	if not os.path.exists(item.get('title')):
		try:
			os.mkdir(item.get('title'))
		except OSError:
			return None
		except FileExistsError:
			pass
	try:
		response = requests.get(item.get('image'))
		if response.status_code == 200:
			file_path = os.path.join(item.get('title'), md5(response.content).hexdigest() + '.jpg')
			if not os.path.exists(file_path):
				with open(file_path, 'wb') as f:
					f.write(response.content)
			else:
				print("Downloaded", file_path)
	except requests.ConnectionError:
		print('Save Failed')
 
 
def main(offset):
	json = get_page(offset)
	for item in get_images(json):
		save_image(item)
 
 
GROUP_START = 0
GROUP_END = 20
if __name__ == '__main__':
	pool = Pool()
	groups = ([x * 20 for x in range(GROUP_START, GROUP_END + 1)])
	pool.map(main, groups)
	pool.close()
	pool.join()