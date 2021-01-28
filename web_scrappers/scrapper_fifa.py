import requests
import re
import pandas as pd
from bs4 import BeautifulSoup

SEASON_TO_PAGES = {
    '05': 17,
    '06': 16,
    '07': 17,
    '08': 17,
    '09': 22,
    '10': 23,
    '11': 23,
    '12': 21,
    '13': 21,
    '14': 22,
    '15': 22,
    '16': 22,
    '17': 22,
    '18': 22,
    '19': 22
}

if __name__ == '__main__':
    list_table = list()
    list_attr = ["name", "nation", "year", "team", "ovr", "pot", "age", "pos_num", "hits"]

    for season in SEASON_TO_PAGES.keys():
        print(season + ' START !')
        for page in range(1, SEASON_TO_PAGES[season] + 1):
            webpage = requests.get('https://www.fifaindex.com/players/fifa' + str(season) + '/'
                                   + str(page) + '/?league=13&order=desc')
            soup = BeautifulSoup(webpage.text, 'html.parser')
            players = soup.find_all('tr', attrs={'data-playerid': re.compile('[1-9]*')})
            for player in players:
                name = player.find(attrs={'data-title': 'Name'}).text
                overal = player.find(attrs={'data-title': 'OVR / POT'}).next.next
                pot = overal.next.next
                age = player.find(attrs={'data-title': 'Age'}).text
                hits = player.find(attrs={'data-title': 'Hits'}).text
                nationality = player.find(attrs={'data-title': 'Nationality'}).next.attrs['title']
                pos_num = len(player.find_all(attrs={'class': 'link-position'}))
                year = season
                team = player.find(attrs={'data-title': 'Team'}).next.attrs['title'].split('FIFA')[0].rstrip()
                list_player = [name, nationality, year, team, overal, pot, age, pos_num, hits]
                list_table.append(list_player)
        print(season + ' FINISH !')

    my_df = pd.DataFrame(list_table, columns=list_attr)
    my_df.to_csv('fifa_players_stats.csv')
