import urllib
from urllib import request
import requests

proxies = {'http' : 'http://127.0.0.1:1087', \
           'https' : 'https://127.0.0.1:1087'}

# proxies = {'http' : 'socks5://127.0.0.1:1086', \
#          'https' : 'socks5://127.0.0.1:1086'}

print(proxies)

"""
proxies_processor = urllib.request.ProxyHandler(proxies)

print(proxies_processor)

opener = urllib.request.build_opener(proxies_processor, sslHandler)

print(opener)
"""

headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36'}

# google_url = 'https://www.google.com'
google_url = 'https://www.github.com'
opener = request.build_opener(request.ProxyHandler(proxies))
request.install_opener(opener)

req = request.Request(google_url, headers = headers)

print(req)

response = request.urlopen(req)

"""
print(response.read().decode())
"""
