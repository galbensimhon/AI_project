import pandas as pd

COUNT_FOUND_SAME_YEAR = 0
COUNT_FOUND_NEXT_YEAR = 0
COUNT_FOUND_PREV_YEAR = 0
COUNT_FOUND = 0
COUNT_NOT_FOUND = 0
CRT = 0

GET_DEFAULT_AVG = 0
REMOVE_LINE = 1

MAKE_GAME_RESULT_ATTR = 0
SET_GAME_RESULTS = 1
SET_DIFF_SCORES = 0
CSV_GAME_RESULT = 1


# TODO: check 3 options for database: (don't forget change name csv)
# for making csv _diff_score - SET_DIFF_SCORES = 1, SET_GAME_RESULTS = 1,  CSV_GAME_RESULT = 0
# for making csv _game_result - SET_DIFF_SCORES = 0, SET_GAME_RESULTS = 1,  CSV_GAME_RESULT = 1,
# MAKE_GAME_RESULT_ATTR = 0
# in case don't find some player:
# GET_DEFAULT_AVG = 0 -> calculate avg team, GET_DEFAULT_AVG = 1 -> set 65 value, REMOVE_LINE = 1 -> delete this line


def prnt(curr_year):
    global CRT
    if CRT != curr_year:
        CRT = curr_year
        print(curr_year)


def update_player_attr_in_list(list_to_append, attr):
    for feature in ['ovr', 'pot', 'nation', 'age', 'pos_num', 'hits']:
        list_to_append.append(attr[feature].values[0])
    return attr['ovr'].values[0]


def get_player_attr(fifa, year, name, count=False):
    global COUNT_FOUND_SAME_YEAR
    global COUNT_FOUND_NEXT_YEAR
    global COUNT_FOUND_PREV_YEAR
    global COUNT_FOUND
    global COUNT_NOT_FOUND

    ret = fifa.loc[(fifa['year'] == year) & (fifa['name'].str.upper() == name)]
    if ret.values.size > 0:
        if count:
            COUNT_FOUND_SAME_YEAR += 1
        return ret

    ret = fifa.loc[(fifa['year'] == year + 1) & (fifa['name'].str.upper() == name)]
    if ret.values.size > 0:
        if count:
            COUNT_FOUND_NEXT_YEAR += 1
        return ret

    ret = fifa.loc[(fifa['year'] == year - 1) & (fifa['name'].str.upper() == name)]
    if ret.values.size > 0:
        if count:
            COUNT_FOUND_PREV_YEAR += 1
        return ret

    ret = fifa.loc[(fifa['name'].str.upper() == name)]
    if ret.values.size > 0:
        if count:
            COUNT_FOUND += 1
        return ret
    if count:
        COUNT_NOT_FOUND += 1
    return ret


def default_update(list_to_append, average_rate=65):
    list_to_append.append(average_rate)  # ovr
    list_to_append.append(average_rate)  # pot
    list_to_append.append('England')  # nation
    list_to_append.append(24)  # age
    list_to_append.append(1)  # pos_num
    list_to_append.append(0)  # hits


def set_player_attr(list_attr, str_player):
    for i in range(1, 12):
        for feature in [' _ovr', '_pot', '_nation', '_age', '_posnum', '_hits']:
            list_attr.append(str_player + str(i) + feature)


def set_team_attr(list_attr, str_team):
    for feature in ['_defense', ' _middle', '_offense']:
        list_attr.append(str_team + feature)


def make_list_attr():
    list_attr = list()
    list_attr.append('home_team')
    set_player_attr(list_attr=list_attr, str_player='hp')
    set_team_attr(list_attr=list_attr, str_team='ht')
    list_attr.append('away_team')
    set_player_attr(list_attr=list_attr, str_player='ap')
    set_team_attr(list_attr=list_attr, str_team='at')
    list_attr.append('season')
    list_attr.append('diff_score')
    if MAKE_GAME_RESULT_ATTR:
        list_attr.append('game_results')
    return list_attr


def get_average_rates(players_name, fifa_file, curr_year):
    count_players_found = 0
    sum_points_players_found = 0
    for pl_name in players_name:
        ret_val = get_player_attr(fifa=fifa_file, year=curr_year, name=pl_name, count=False)
        if ret_val.values.size > 0:
            rate = ret_val['ovr'].values[0]
            sum_points_players_found += rate
            count_players_found += 1
        elif REMOVE_LINE:
            return 0, False
    return sum_points_players_found / count_players_found if count_players_found != 0 else 0, len(players_name) == \
           count_players_found


def set_team_line(list_to_update, players_name, fifa_file, curr_year, by_average):
    fifa_defense = 0
    fifa_center = 0
    fifa_offence = 0
    i = 0
    for pl_name in players_name:
        rate = 65  # default
        ret_val = get_player_attr(fifa=fifa_file, year=curr_year, name=pl_name, count=True)
        if ret_val.values.size > 0:
            rate = update_player_attr_in_list(list_to_append=list_to_update, attr=ret_val)
        else:
            default_update(list_to_append=list_to_update, average_rate=by_average)
        if i < 5:  # at soccer we have 5 defense players
            fifa_defense += rate
        elif i < 8:  # at soccer we have 3 center players
            fifa_center += rate
        else:  # at soccer we have 3 offence players
            fifa_offence += rate

        i += 1

    list_to_update.append(fifa_defense / 5)
    list_to_update.append(fifa_center / 3)
    list_to_update.append(fifa_offence / 3)


def make_teams_list(row):
    list_home = list()
    list_away = list()
    for i in range(1, 12):
        home_name = row['hp' + str(i)].upper()
        away_name = row['ap' + str(i)].upper()
        list_away.append(away_name)
        list_home.append(home_name)
    return list_home, list_away


def print_for_debug():
    print("fount next year is " + str(COUNT_FOUND_SAME_YEAR) + ' team_lineups.csv')
    print("fount next year is " + str(COUNT_FOUND_NEXT_YEAR) + ' team_lineups.csv')
    print("fount prev year is " + str(COUNT_FOUND_PREV_YEAR) + ' team_lineups.csv')
    print("fount some year is " + str(COUNT_FOUND) + ' team_lineups.csv')
    print("no found  is " + str(COUNT_NOT_FOUND) + ' team_lineups.csv')


if __name__ == '__main__':

    fifa = pd.read_csv('fifa_players_stats.csv', engine='python')
    matches = pd.read_csv('team_lineups.csv', engine='python')
    list_all = list()
    list_attr = make_list_attr()
    bad_row_count = 0
    to_delete = True

    for match in matches.iterrows():
        row = match[1]
        list_per_match = list()
        year = row['season']
        home_players_names, away_players_names = make_teams_list(row)
        avg_rates_home, found_all_home = get_average_rates(players_name=home_players_names, fifa_file=fifa,
                                                           curr_year=year)
        avg_rates_away, found_all_away = get_average_rates(players_name=away_players_names, fifa_file=fifa,
                                                           curr_year=year)
        if GET_DEFAULT_AVG:
            avg_rates_home = 65
            avg_rates_away = 65

        if REMOVE_LINE:
            if not found_all_home or not found_all_away:
                prnt(curr_year=year)
                continue
        if not found_all_home or not found_all_away:
            bad_row_count += 1
        list_per_match.append(row['home_team'].upper())

        set_team_line(list_to_update=list_per_match, players_name=home_players_names, fifa_file=fifa, curr_year=year,
                      by_average=avg_rates_home)

        list_per_match.append(row['away_team'].upper())
        set_team_line(list_to_update=list_per_match, players_name=away_players_names, fifa_file=fifa, curr_year=year,
                      by_average=avg_rates_away)
        list_per_match.append(row['season'])

        # the next attr is the classification
        if SET_DIFF_SCORES:
            list_per_match.append(row['diff_score'])
        if SET_GAME_RESULTS:
            list_per_match.append(row['game_results'])
        prnt(curr_year=year)
        list_all.append(list_per_match)

    my_df = pd.DataFrame(list_all, columns=list_attr)
    if CSV_GAME_RESULT:
        my_df["diff_score"] = my_df["diff_score"].replace("2", -1)

    print_for_debug()
    print("bad row is " + str(bad_row_count))
    my_df.to_csv('combine_with_delete_game_results.csv')
