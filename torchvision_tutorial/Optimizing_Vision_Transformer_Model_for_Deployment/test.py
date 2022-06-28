import requests

proxies = {"http" : "socks5://127.0.0.1:1086", \
           "https": "socks5://127.0.0.1:1086"}

res = requests.get('http://www.github.com/', proxies = proxies)
print(res.text)
