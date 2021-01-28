import requests
import pandas as pd
from bs4 import BeautifulSoup


SEASON_TO_URL = {
    '5': 5188,
    '6': 5568,
    '7': 5947,
    '8': 6237,
    '9': 6707,
    '10': 7087,
    '11': 7467,
    '12': 7864,
    '13': 9231,
    '14': 9611,
    '15': 12115,
    '16': 14040,
    '17': 22342,
    '18': 38308,
    '19': 46605
}


if __name__ == '__main__':
    list_table = list()
    list_attr = ["home_team", "hp1", "hp2", "hp3", "hp4", "hp5", "hp6", "hp7", "hp8", "hp9", "hp10", "hp11",
                 "away_team", "ap1", "ap2", "ap3", "ap4", "ap5", "ap6", "ap7", "ap8", "ap9", "ap10", "ap11",
                 "home_score", "away_score", "diff_score", "game_results", "season"]
    for season in SEASON_TO_URL.keys():
        print(season + ' START !')
        for i in range(379):
            suffix = i + SEASON_TO_URL[season]
            list_home = list()
            list_away = list()
            webpage = requests.get('https://www.premierleague.com/match/' + str(suffix))
            soup = BeautifulSoup(webpage.text, 'html.parser')
            names = soup.find_all(attrs={'class': 'squadHeader'})
            home_name = names[0].find(attrs={'class': 'position'}).contents[0].string[21:].rstrip()
            away_name = names[1].find(attrs={'class': 'position'}).contents[0].string[21:].rstrip()
            list_home.append(home_name)
            list_away.append(away_name)
            match = soup.find_all(attrs={'class': 'matchLineupTeamContainer'})
            home_team = match[0].find_all(attrs={'class': 'info'})
            away_team = match[1].find_all(attrs={'class': 'info'})
            for j in range(11):
                list_home.append(home_team[j].contents[1].contents[0].string[25:].rstrip())
                list_away.append(away_team[j].contents[1].contents[0].string[25:].rstrip())
            list_ret = list_home + list_away
            score = soup.find(title=home_name).parent.parent.find(attrs={'class': 'score'}).text.split('-')
            home_goal = score[0]
            away_goal = score[1]
            game_diff = int(home_goal) - int(away_goal)
            game_res = 0
            if game_diff > 0:
                game_res = 1
            elif game_diff < 0:
                game_res = -1
            list_ret.append(home_goal)
            list_ret.append(away_goal)
            list_ret.append(game_diff)
            list_ret.append(game_res)
            list_ret.append(season)
            list_table.append(list_ret)
        print(season + ' FINISH !')
    my_df = pd.DataFrame(list_table, columns=list_attr)
    my_df.to_csv('team_lineups.csv')
