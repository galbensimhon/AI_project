import pandas as pd

count = 0
CRT = 0


def prnt(curr_year):
    global CRT
    if CRT != curr_year:
        CRT = curr_year
        print(curr_year)


def check_name(fifa_l, name):
    ret = fifa_l.loc[(fifa_l['name'].str.upper() == name)]
    if ret.values.size > 0:
        return False

    return True


def check_players(mising_list, players_name, fifa_file):
    global count
    for pl_name in players_name:
        ret_val = check_name(fifa_l=fifa_file, name=pl_name)
        if ret_val:
            count += 1
            mising_list.append(pl_name)


if __name__ == '__main__':

    fifa = pd.read_csv('fifa_players_stats.csv', engine='python')
    matchs = pd.read_csv('team_lineups.csv', engine='python')
    list_per_match = list()
    for match in matchs.iterrows():
        raw = match[1]
        year = raw['season']
        Away_players_names = list()
        Home_players_names = list()
        for i in range(1, 12):
            Away_name = raw['ap' + str(i)].upper()
            Home_name = raw['hp' + str(i)].upper()
            Away_players_names.append(Away_name)
            Home_players_names.append(Home_name)

        check_players(mising_list=list_per_match, players_name=Home_players_names, fifa_file=fifa)
        check_players(mising_list=list_per_match, players_name=Away_players_names, fifa_file=fifa)
        prnt(curr_year=year)

    mis = [[name, list_per_match.count(name)] for name in set(list_per_match)]
    mis_df = pd.DataFrame(mis)
    mis_df.to_csv('missing_after_new.csv')
    print("missing" + str(count))
