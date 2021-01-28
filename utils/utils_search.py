import random

from sklearn.feature_selection import SelectKBest, f_regression, f_classif

from utils.class_and_reg import *

GK = 0
DEF = 1
CM = 2
WING = 3
ATT = 4
NUM_OF_STEP = 7
USE_ADVANCED_LOCAL_SEARCH = 1

MAX_SEARCH_STEPS = 20
PREC = 0.9

GAMES_FOR_AVERAGE = {}

SHOW_GRAPH = 1
SHOW_GRAPH_DBG = 1


def create_dict_games(teams):
    """
    create dictionary of all games to average in future
    :param teams: the databases which includes all teams
    :return: dict
    """
    for key_home in teams.keys():
        if key_home == 'Unnamed: 0':
            continue
        for key_away in teams.keys():
            if key_home != key_away and key_away != 'Unnamed: 0':
                GAMES_FOR_AVERAGE[key_home] = key_away


class Player:
    """
    class of Player, which includes his attributes: name, nation, overal, potential, age, position's num, hits, position
    """
    def __init__(self, name=None, nation=None, overal=None, pot=None, age=None, pos_num=None, hits=None, pos=None):
        if name is None or nation is None or overal is None or pot is None or age is None or pos_num is None \
                or hits is None or pos is None:
            print('Player initial does not consist illegal params')
            exit(-1)
        self.name = name
        self.nation = nation
        self.overal = int(overal)
        self.pot = int(pot)
        self.age = int(age)
        self.pos_num = int(pos_num)
        self.hits = int(hits)
        self.pos = int(pos)

    def to_list(self, list_to_append: list):
        list_to_append.append(NATAION_TO_RANKING[self.nation])
        list_to_append.append(self.overal)
        list_to_append.append(self.pot)
        list_to_append.append(self.age)
        list_to_append.append(self.pos_num)
        list_to_append.append(self.hits)


class Team:
    """
    class of Team, which includes his attributes: name, and sets of: goalkeeper, defense, center, wings, attack
    line-up for team must be 4-4-2 (4 player in defense, 4 players in center, 2 in attack
    """
    def __init__(self, name, goalkeeper: set = None, defense: set = None, center: set = None, wings: set = None,
                 attack: set = None):
        if name is None or goalkeeper is None or len(goalkeeper) < 1 or defense is None or len(defense) < 4 or \
                center is None or len(center) < 2 or wings is None or len(wings) < 2 or attack is None or \
                len(attack) < 2:
            print('Team initial does not consist illegal lineup')
            exit(-1)
        self.name = name
        self.goalkeeper = goalkeeper
        self.defense = defense
        self.center = center
        self.wings = wings
        self.attack = attack

    def to_list(self, list_to_append: list):
        list_to_append.append(self.name)


def create_team(team: dict, name: str):
    """
    create Team object from database : team with name
    :param team: the database which includes all players from team
    :param name: the name of specific team
    :return: Team object
    """
    goal_keeper = set()
    defense = set()
    center = set()
    wings = set()
    attack = set()
    i = 0
    while i + 7 <= len(team):
        player = Player(name=team[i], nation=team[i+1], overal=team[i+2], pot=team[i+3], age=team[i+4],
                        pos_num=team[i+5], hits=team[i+6], pos=team[i+7])
        if int(player.pos) == GK:
            goal_keeper.add(player)
        elif int(player.pos) == DEF:
            defense.add(player)
        elif int(player.pos) == CM:
            center.add(player)
        elif int(player.pos) == WING:
            wings.add(player)
        elif int(player.pos) == ATT:
            attack.add(player)
        else:
            print('Something Wrong')
            exit(-1)
        i += 8

    return Team(name=name, goalkeeper=goal_keeper, defense=defense, center=center, wings=wings, attack=attack)


def get_random_player(temp_set):
    return random.sample(temp_set, 1)[0]


def get_max_player_ovr(temp_set):
    """
    return the max player by his overal from set
    :param temp_set: the set which includes the players
    :return: Player object
    """
    max_player = None
    for player in temp_set:
        if max_player is None:
            max_player = player
        elif player.overal > max_player.overal:
            max_player = player

    return max_player


def append_to_list(list_lineup: list, set_player: set, num_of_players: int, get_best: bool):
    """
    append some numbers of players to list lineup
    :param list_lineup: the list to insert
    :param set_player: the set which includes possible players to insert
    :param num_of_players: the number of players to append to list
    :param get_best: if True - append the best player by overal, if False - append random player
    """
    temp_set = set_player.copy()
    temp_list = list()

    while len(temp_list) < num_of_players:
        max_player = get_max_player_ovr(temp_set) if get_best else get_random_player(temp_set)
        temp_list.append(max_player)
        temp_set.remove(max_player)

    for item in temp_list:
        list_lineup.append(item)


def make_xi(team: Team, get_best: bool):
    """
    create list of 11 lineups player to xi list
    :param team: the database of team's players
    :param get_best: if True - get the best player by overal, if False - get random player
    :return: list to classify/regression algorithm
    """
    list_best_xi = list()
    append_to_list(list_best_xi, team.goalkeeper, 1, get_best=get_best)
    append_to_list(list_best_xi, team.defense, 4, get_best=get_best)
    append_to_list(list_best_xi, team.center, 2, get_best=get_best)
    append_to_list(list_best_xi, team.wings, 2, get_best=get_best)
    append_to_list(list_best_xi, team.attack, 2, get_best=get_best)

    return list_best_xi


def xi_to_list(team_name: str, list_players: list):
    """
    return the final list for classify/regression algorithm
    :param team_name: the name of team
    :param list_players: includes the team lineup
    :return: the final list for classify/regression algorithm
    """
    list_to_return = list()
    list_to_return.append(TEAM_TO_RANKING[team_name])
    fifa_defense = 0
    fifa_center = 0
    fifa_attack = 0
    for player in list_players:
        player.to_list(list_to_return)
        if player.pos == GK or player.pos == DEF:
            fifa_defense += player.overal
        elif player.pos == CM or player.pos == WING:
            fifa_center += player.overal
        elif player.pos == ATT:
            fifa_attack += player.overal

    list_to_return.append(fifa_defense/5)
    list_to_return.append(fifa_center/4)
    list_to_return.append(fifa_attack/2)

    return list_to_return


def xi_to_players_names(list_players: list):
    """
    return from list of Player object, list of their names
    :param list_players: list which includes players
    :return: list of names
    """
    list_to_return = list()
    for player in list_players:
        list_to_return.append(player.name)

    return list_to_return


def get_all_players_by_pos(xi: list, pos: int):
    """
    return list of all players with specific pos
    :param xi: the database of all players
    :param pos: the specific position
    :return: list of players with same position
    """
    list_to_return = list()
    for player in xi:
        if player.pos == pos:
            list_to_return.append(player)
    return list_to_return


def set_pos_player(xi: list, index: int, team: Team):
    """
    return set of all players which aren't in lineup
    :param xi: the database of team's players
    :param index: the index of player in team' lineup
    :param team: the team Object
    :return: set of all players which aren't in lineup
    """
    pos = xi[index].pos
    without_players = get_all_players_by_pos(xi=xi, pos=pos)
    temp_set = None
    if pos == GK:
        temp_set = team.goalkeeper.copy()
    elif pos == DEF:
        temp_set = team.defense.copy()
    elif pos == CM:
        temp_set = team.center.copy()
    elif pos == WING:
        temp_set = team.wings.copy()
    elif pos == ATT:
        temp_set = team.attack.copy()

    for player in without_players:
        temp_set.remove(player)

    return temp_set


def get_player(team: Team, xi: list, index: int, get_best: bool):
    """
    return player from team which isn't in lineup
    :param team: the team's players
    :param xi: the lineup of team
    :param index: the index of player with specific position
    :param get_best:
    :return: player from team which isn't in lineup
    """
    temp_set = set_pos_player(xi=xi, index=index, team=team)
    if len(temp_set) > 0:
        player = get_max_player_ovr(temp_set=temp_set) if get_best else get_random_player(temp_set=temp_set)
    else:
        player = xi[index]
    return player


def change_xi(xi: list, index: int, team: Team, get_best: bool):
    """
    change specific player with index from team
    :param xi: the lineup
    :param index: the specific index of player
    :param team: the team's players
    :param get_best: if True - get the best player by overal, if false - get random
    :return: new list with the changing
    """
    temp_list = xi.copy()
    temp_list[index] = get_player(team=team, xi=xi, index=index, get_best=get_best)
    return temp_list


def change_xi_with_player(xi: list, index: int, player: Player):
    """
    return new list with new player
    :param xi: the lineup
    :param index: the index of player to change
    :param player: the new player to insert
    :return: new list with new player
    """
    tmp_list = xi.copy()
    tmp_list[index] = player
    return tmp_list


def select_k_best_features(train, score_func, k):
    train_data = train[train.columns.values[:-1]]
    train_target = train[train.columns.values[-1]]
    best_features = SelectKBest(score_func=score_func, k=k)
    fit = best_features.fit(train_data, train_target)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(train_data.columns)
    feature_scores = pd.concat([df_columns, df_scores], axis=1)
    feature_scores.columns = ['Specs', 'Score']  # naming the dataframe columns
    ret_list = list(feature_scores.nlargest(k, 'Score')['Specs'])
    feature_scores.nlargest(k, 'Score')
    ret_list.append('diff_score')
    return ret_list


def init_clf():
    """
    initial classifier to predict
    """
    combine = pd.read_csv('databases/combine_databases/combine_game_results.csv', engine='python')
    combine = combine[combine.columns.values[2:]]
    list_attr = combine.columns.values[:-1]
    combine = prepare_label(combine)
    clf_index = select_k_best_features(train=combine, score_func=f_classif, k=80)
    combine = combine[clf_index]
    train_data = combine[combine.columns.values[:-1]]
    train_target = combine['diff_score']
    clf = RandomForestClassifier(n_estimators=RF_N_EST_CLF, max_depth=RF_MAX_DEPTH_CLF)
    clf.fit(train_data, train_target)

    return clf, list_attr, clf_index


def init_reg():
    """
    initial regressor to predict
    """
    combine = pd.read_csv('databases\combine_databases\combine_with_delete_diff_score.csv', engine='python')
    combine = combine[combine.columns.values[1:-1]]
    list_attr = combine.columns.values[:-1]
    combine = prepare_label(combine)
    reg_index = select_k_best_features(train=combine, score_func=f_regression, k=80)
    combine = combine[reg_index]
    train_data = combine[combine.columns.values[:-1]]
    train_target = combine['diff_score']
    clf = RandomForestRegressor(n_estimators=RF_N_EST_REG, max_depth=RF_MAX_DEPTH_REG)
    clf.fit(train_data, train_target)

    return clf, list_attr, reg_index
