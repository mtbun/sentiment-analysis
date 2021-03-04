import requests
import json
import re
import time

class Twitter(object):
	def __init__(self, session):
		self.session = session

	def search(self, query, count=20, cursor=None):
		url = 'https://twitter.com/i/api/2/search/adaptive.json'
		params = {
			"q": query,
			"tweet_search_mode": "live",
			"count": count,
			"query_source": "typed_query",
			"cursor": cursor,
			"include_ext_alt_text": True
		}
		return self.session.get(url, params=params)


	def find_cursor(self, string):
		results = re.findall(r'"value"\s*:\s*"(scroll:.+?)"', string)
		if len(results):
			return results[0]
		print('[!] Cursor not found')
		return None

# my cookie.
# validity period one year
session = requests.Session()
session.headers.update(
	{
		'cookie': '_ga=GA1.2.1495288742.1613249546; _gid=GA1.2.334880179.1614500714; external_referer=padhuUp37zjgzgv1mFWxJ8aHbAM%2FyKh7|0|8e8t2xd8A2w%3D; twid=u%3D1164272513611948032; lang=en; ct0=96c1989a05c6fdfcd4f11abc5f7d7ba6c67e826650c9c4bb1bd22a3f6a392aba5012dab77753e8ac3537e6783fa7c01345ade9e74d099029939bbfd303a24af18bc3e196f02a5e1cd07075ea10c229a9; ads_prefs="HBERAAA="; auth_token=8eb4bfe71daf3d1782d87617a461b6bda96a8744; kdt=Wn7ceJCXfLvipbsZtqBXS0gUN7P7loF2ePku6DOm; remember_checked_on=1; guest_id=v1%3A161323755113127861; personalization_id="v1_FYelPDiMEi+gkNU0jem/OQ=="',
		'authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA',
		'user-agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:85.0) Gecko/20100101 Firefox/85.0',
		'x-csrf-token': '96c1989a05c6fdfcd4f11abc5f7d7ba6c67e826650c9c4bb1bd22a3f6a392aba5012dab77753e8ac3537e6783fa7c01345ade9e74d099029939bbfd303a24af18bc3e196f02a5e1cd07075ea10c229a9',
		#'x-guest-token': '1366808088741900288'
        #no need if my cookie saved after login
	}
)


twitter = Twitter(session)

#################################################################################
query = 'exciting'
cursor = None


for _ in range(50):
	response = twitter.search(query=query, count=100, cursor=cursor)
	try:
		tweets = response.json()['globalObjects']['tweets']
	except KeyError:
		print(response.json())
		exit()

	for k in tweets.keys():
		try:
			tweet = tweets[k]
			text = tweet['text']
			text = text.replace(',', '').replace('"', '').replace('\n', ' ')
			with open(f'{query}.csv', 'a+') as f:
				f.write(f'"{text}",\n')
		except KeyError:
			print('[i] No text.')

	print(len(tweets), _ + 1)
	cursor = twitter.find_cursor(response.text)
	time.sleep(10)