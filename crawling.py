import requests
from bs4 import BeautifulSoup

###################################################################
# AI 관련 기사 중 제목, 언론사, 링크주소를 크롤링하는 함수
def crawl_titles_presses_links():
  headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'}
  url = f'https://search.naver.com/search.naver?where=news&sm=tab_jum&query=AI'
  response = requests.get(headers=headers, url=url)

  soup = BeautifulSoup(response.text, 'html.parser')
  info_list = soup.select('.info_group')
  
  titles = []
  presses = []
  links = []
  for i in range(len(info_list)):
    press = info_list[i].select_one('a').text
    if press[-6:] == '언론사 선정':
      press = press[:-6]
    title = soup.select('.news_contents')[i].select_one('.news_tit').text
    try:
      link = info_list[i].select('a')[1].attrs['href']
    except:
      continue
    
    titles.append(title)
    presses.append(press)
    links.append(link)
    
  return titles, presses, links

###################################################################
# 링크에 들어가 기사를 추출하는 함수
def extract_news(link):
  headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'}
  url = link
  response = requests.get(headers=headers, url=url)
  soup = BeautifulSoup(response.text, 'html.parser')
  news = soup.select_one('#dic_area').text
  date = soup.select_one('._ARTICLE_DATE_TIME').attrs['data-date-time']
  return news, date
#################################

#테스트용
#################################
def crawl_news():
  # requests 모듈을 통해서 요청보내고, html 문서받기
  headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'}
  url = f'https://n.news.naver.com/mnews/article/001/0014400079?sid=101' ##### 예시 >> 수정 필요
  response = requests.get(headers=headers, url=url)

  # 파싱하기
  soup = BeautifulSoup(response.text, 'html.parser')

  # 원하는 정보 선택하기
  paragraphs = soup.select_one('._ARTICLE_DATE_TIME').attrs['data-date-time']
  
  news = []
  for paragraph in paragraphs:
    news.append(paragraph.strip())
  news = ' '.join(news)
  return paragraphs

# 1. 기사링크와 언론사를 return하는 함수
# 2. 언론사마다 기사가 들어 있는 부분을 return하는 함수
# print(crawl_news())