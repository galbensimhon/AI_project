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
    '19': 22,
}

GK = 0
DEF = 1
CM = 2
WING = 3
ATT = 4


def pos_to_num(pos_is):
    if pos_is in gk_list:
        return GK
    if pos_is in def_list:
        return DEF
    if pos_is in cm_list:
        return CM
    if pos_is in wing_list:
        return WING
    if pos_is in attck_list:
        return ATT
    else:
        print(pos_is)
        return 5


if __name__ == '__main__':
    list_table = list()
    cm_list = ['CAM', 'CDM', 'CM', 'LAM', 'LCAM', 'LCDM', 'LCM', 'LDM', 'RAM', 'RCAM', 'RCDM', 'RCM', 'RDM']
    def_list = ['CB', 'LB', 'LCB', 'LWB', 'RB', 'RCB', 'RWB', 'SW']
    wing_list = ['LWM', 'LM', 'RM', 'RWM', 'RW', 'LW']
    gk_list = ['GK']
    attck_list = ['CF', 'LCF', 'LF', 'LS', 'RCF', 'RF', 'RF', 'ST']
    list_attr = ["name", "nation", "year", "team", "ovr", "pot", "age", "pos_num", "hits"]
    teams_list = ['Everton', 'Sheffield United', 'Leicester City', 'Chelsea', 'Liverpool', 'Southampton', 'Norwich City',
                 'Burnley', 'Manchester United', 'Crystal Palace', 'Wolverhampton Wanderers', 'West Ham United',
                 'Tottenham Hotspur', 'Manchester City', 'Arsenal', 'AFC Bournemouth', 'Brighton & Hove Albion',
                 'Watford', 'Newcastle United', 'Aston Villa']
    team_dict = dict()

    for team in teams_list:
         team_dict[team] = list()

    for page in range(1, 23):
        webpage = requests.get('https://www.fifaindex.com/players/'
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
            pos_is = player.find(attrs={'class': 'link-position'}).text
            pos = pos_to_num(pos_is)
            year = "20"
            team = player.find(attrs={'data-title': 'Team'}).next.attrs['title'].split('FIFA')[0].rstrip()
            list_player = [name, nationality, overal, pot, age, pos_num, hits, pos]
            team_dict[team] = team_dict[team] + list_player
            list_table.append(list_player)

    maxlen = max([len(team_dict[key]) for key in team_dict.keys()])
    print(maxlen)
    [team_dict[key].extend([""] * (maxlen - len(team_dict[key]))) for key in team_dict.keys()]
    print(maxlen)
    new_df = pd.DataFrame(team_dict)
    new_df.to_csv('team_2020_lineups.csv')
