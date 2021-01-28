import sys

from utils.utils_search import *
from utils.graphs import *

# TODO: send by command line 2 options:
#  2 teams as parameters (first team - we represent, second team - the rival)
#  for example: python.exe ./search_lineups.py Everton Chelsea.
#  in this case we represent Everton, and the rival - Chelsea
#  empty parameters - get 20 matches and compare between all search's algorithms

# TODO: SHOW_GRAPH = 1, SHOW_GRAPH_DBG = 1 (graph per game if the user want to show)


def heuristic(reg, clf, tmp_xi: list, clf_index, reg_index, name, list_attr, list_attr2, away_list):
    team_list = xi_to_list(team_name=name, list_players=tmp_xi)
    match = team_list + away_list
    match.append(20)
    tmp = list()
    tmp.append(match)
    clf_df = pd.DataFrame(tmp, columns=list_attr)
    reg_df = pd.DataFrame(tmp, columns=list_attr2)
    match_clf = clf_df[clf_index[:-1]].values
    match_reg = reg_df[reg_index[:-1]].values

    clas = clf.predict(match_clf)
    regs = reg.predict(match_reg)

    return PREC*clas[0] + (1-PREC)*regs[0]


def local_search_advanced(team_home, name_home_team, away_list, classifier, list_attr, clf_index, regressor, list_attr2,
                          reg_index, home_xi):
    list_steps = list()
    cur_max = -np.inf
    flag = True
    i = 0
    m = 0
    # keep checking if we make improvement but not more then MAX steps
    while flag:
        flag = False
        max_xi = home_xi
        # in every step check all position to change...
        for j in range(11):
            pos_set = set_pos_player(xi=home_xi, index=j, team=team_home)
            for player in pos_set:
                m = m + 1
                tmp_xi = change_xi_with_player(xi=home_xi, index=j, player=player)
                fitness = heuristic(clf=classifier, reg=regressor, tmp_xi=tmp_xi, clf_index=clf_index,
                                    reg_index=reg_index, list_attr=list_attr, list_attr2=list_attr2,
                                    name=name_home_team, away_list=away_list)

                if fitness >= cur_max:
                    flag = True
                    cur_max = fitness
                    max_xi = tmp_xi

        home_xi = max_xi
        i = i + 1
        list_steps.append(cur_max)
        if i >= MAX_SEARCH_STEPS:
            break
    ret = xi_to_players_names(list_players=home_xi)
    return cur_max, list_steps, ret, m


def local_search(team_home, name_home_team, away_list, classifier, list_attr, clf_index, regressor, list_attr2,
                 reg_index, home_xi):
    """
    local search algorithm which change player by specific one, and don't check all others
    :param team_home: home team
    :param name_home_team: name of home team
    :param away_list: away team
    :param classifier: classify
    :param list_attr: attributes for classify
    :param clf_index: index for classify
    :param regressor: regressor
    :param list_attr2: attributes for regressor
    :param reg_index: index of regressor 
    :param home_xi: the lineups of home team
    :return: the score of specific lineup of team
    """
    list_steps = list()
    cur_max = -np.inf
    flag = True
    i = 0
    m = 0
    # keep checking if we make improvement but not more then MAX steps
    while flag:
        flag = False
        max_xi = home_xi
        # in every step check all position to change...
        for j in range(11):
            m = m + 1
            tmp_xi = change_xi(xi=home_xi, index=j, team=team_home, get_best=True)
            fitness = heuristic(clf=classifier, reg=regressor, tmp_xi=tmp_xi, clf_index=clf_index, reg_index=reg_index,
                                list_attr2=list_attr2, list_attr=list_attr, name=name_home_team, away_list=away_list)
            if fitness >= cur_max:
                flag = True
                cur_max = fitness
                max_xi = tmp_xi

        home_xi = max_xi
        i = i + 1
        list_steps.append(cur_max)
        if i >= MAX_SEARCH_STEPS:
            break
    ret = xi_to_players_names(list_players=home_xi)
    return cur_max, list_steps, ret, m


def random_restart_search(team_home, name_home_team, away_list, classifier, list_attr, clf_index, regressor, list_attr2,
                          reg_index):
    """
    algorithm search for specific game (2 teams as input)
    :param team_home: home team
    :param name_home_team: name of home team
    :param away_list: away team
    :param classifier: classify
    :param list_attr: attributes for classify
    :param clf_index: index for classify
    :param regressor: regressor
    :param list_attr2: attributes for regressor
    :param reg_index: index of regressor
    :return: lists of steps and fitness for each algorithm
    """
    list_all = list()
    list_fitness = list()
    j = 0
    max_all = -np.inf
    max_xi = []
    search_func = local_search_advanced if USE_ADVANCED_LOCAL_SEARCH else local_search
    while j < NUM_OF_STEP:
        home_xi = make_xi(team_home, (j == 0))
        fitness, list_step, ret_xi, num_step = search_func(team_home=team_home, name_home_team=name_home_team, away_list=away_list,
                                                 classifier=classifier, clf_index=clf_index, regressor=regressor,
                                                 reg_index=reg_index, list_attr=list_attr, list_attr2=list_attr2,
                                                 home_xi=home_xi)
        if fitness > max_all:
            max_all = fitness
            max_xi = ret_xi

        list_all.append(list_step)
        list_fitness.append(fitness)
        j = j + 1

    print("The best lineup:")
    print(max_xi)
    return list_all, list_fitness


def random_restart_search_test(team_home, name_home_team, away_list, classifier, list_attr, clf_index, regressor,
                               list_attr2, reg_index):
    """
    algorithm search for average 20 games
    :param team_home: home team
    :param name_home_team: name of home team
    :param away_list: away team
    :param classifier: classify
    :param list_attr: attributes for classify
    :param clf_index: index for classify
    :param regressor: regressor
    :param list_attr2: attributes for regressor
    :param reg_index: index of regressor
    :return: lists of steps and fitness for each algorithm
    """
    list_all = list()
    list_fitness = list()
    list_all_adv = list()
    list_fitness_adv = list()
    j = first_step = steps = steps_adv = 0
    max_all = -np.inf
    max_all_adv = -np.inf
    while j < NUM_OF_STEP:
        home_xi = make_xi(team_home, (j == 0))
        fitness, list_step, ret_xi, num_steps = local_search(team_home=team_home, name_home_team=name_home_team,
                                                  away_list=away_list, classifier=classifier, clf_index=clf_index,
                                                  regressor=regressor, reg_index=reg_index, list_attr=list_attr,
                                                  list_attr2=list_attr2, home_xi=home_xi)
        if fitness > max_all:
            max_all = fitness

        list_all.append(list_step)
        list_fitness.append(fitness)
        steps = steps + num_steps
        fitness_adv, list_step_adv, ret_xi_adv, num_steps_adv = local_search_advanced(team_home=team_home, name_home_team=name_home_team,
                                                                       away_list=away_list, classifier=classifier,
                                                                       clf_index=clf_index, regressor=regressor,
                                                                       reg_index=reg_index, list_attr=list_attr,
                                                                       list_attr2=list_attr2, home_xi=home_xi)
        if fitness_adv > max_all_adv:
            max_all_adv = fitness_adv

        if j == 0:
            first_step = num_steps_adv

        list_all_adv.append(list_step_adv)
        list_fitness_adv.append(fitness_adv)
        steps_adv = steps_adv + num_steps_adv

        j = j + 1

    return list_all, list_fitness, steps, list_all_adv, list_fitness_adv, steps_adv, first_step


def prepare_teams(name_home_team, name_away_team, teams):
    """
    create list of home and away team
    :param name_home_team: name home team
    :param name_away_team: name away team
    :param teams: databases which includes all teams
    :return: home and away team
    """

    if name_home_team is None or name_away_team is None:
        print('Illegal name teams was provided')
        exit(-1)

    home_team = teams[name_home_team] if name_home_team in teams.keys() else None
    away_team = teams[name_away_team] if name_away_team in teams.keys() else None

    if home_team is None or away_team is None:
        print('Illegal teams was provided')
        print('The teams are:')
        print(teams.keys())
        exit(-1)

    team_home = create_team(dict(home_team), name_home_team)
    team_away = create_team(dict(away_team), name_away_team)

    return team_home, team_away


def search(name_home_team, name_away_team, teams_2020, classifier, list_attr, clf_index, regressor, list_attr2,
           reg_index, get_average: bool):
    """
    create searching for best team lineup and print it
    :param name_home_team: the home team which we represent
    :param name_away_team: the rival
    :param teams_2020: the databases which includes all teams
    :param classifier: classify
    :param list_attr: attributes for classify
    :param clf_index: index for classify
    :param regressor: regressor
    :param list_attr2: attributes for regressor
    :param reg_index: index of regressor
    :param get_average: boolean for average or not

    :return: attributes in case of average
    """
    team_home, team_away = prepare_teams(name_home_team=name_home_team, name_away_team=name_away_team,
                                         teams=teams_2020)
    away_xi = make_xi(team_away, True)
    away_list = xi_to_list(team_name=team_away.name, list_players=away_xi)
    random_restart = random_restart_search_test if get_average is True else random_restart_search

    if get_average:
        list_all, list_fitness, steps, list_all_adv, list_fitness_adv, steps_adv, first_step = \
            random_restart(team_home=team_home, name_home_team=team_home.name, away_list=away_list,
                           classifier=classifier, clf_index=clf_index, regressor=regressor, reg_index=reg_index,
                           list_attr=list_attr, list_attr2=list_attr2)
        if SHOW_GRAPH and SHOW_GRAPH_DBG:
            show_graph_search_graph(list_all=list_all)
            show_graph_search_graph(list_all=list_all_adv)
            show_graph_fitness_bar(list_all=list_fitness, list_adv=list_fitness_adv)

        return list_fitness[0], len(list_all[0]) * 11, max(list_fitness), steps, list_fitness_adv[0], first_step, \
               max(list_fitness_adv), steps_adv, list_all[0][0]
    else:
        list_all, list_fitness = \
            random_restart(team_home=team_home, name_home_team=team_home.name, away_list=away_list,
                           classifier=classifier,
                           clf_index=clf_index, regressor=regressor, reg_index=reg_index, list_attr=list_attr,
                           list_attr2=list_attr2)
        if SHOW_GRAPH and SHOW_GRAPH_DBG:
            show_graph_search_graph(list_all=list_all)
            show_graph_fitness_bar_one_list(list_all=list_fitness)


if __name__ == '__main__':

    classifier, list_attr, clf_index = init_clf()
    regressor, list_attr2, reg_index = init_reg()
    teams_2020 = pd.read_csv('databases/raw_databases/team_2020_lineups.csv', engine='python')
    num_of_params = len(sys.argv)
    # the user can select specific game by home team and away team as input,
    # or get average of 20 random games
    name_home_team = str(sys.argv[1]) if num_of_params >= 2 else None
    name_away_team = str(sys.argv[2]) if num_of_params >= 3 else None

    if name_home_team and name_away_team:
        search(name_home_team=name_home_team, name_away_team=name_away_team,
               teams_2020=teams_2020, classifier=classifier, list_attr=list_attr, clf_index=clf_index,
               regressor=regressor, list_attr2=list_attr2, reg_index=reg_index, get_average=False)
    elif name_away_team or name_home_team:
        print('Please select 2 names of teams, or none of them')
        exit(-1)
    else:
        create_dict_games(teams=teams_2020)
        sum_fitness_max = sum_steps_max = sum_fitness_rs = sum_steps_rs = sum_fitness_max_adv = sum_steps_max_adv = \
            sum_fitness_rs_adv = sum_steps_rs_adv = 0
        sum_fitness_max_xi = fitness_max_xi = 0
        len_games = len(GAMES_FOR_AVERAGE.keys())
        for key in GAMES_FOR_AVERAGE.keys():
            fitness_by_max_initial_team, steps_max, fitness_by_rs_initial, steps_rs, fitness_by_max_initial_team_adv, steps_max_adv, \
            fitness_by_rs_initial_adv, steps_rs_adv, fitness_max_xi = search(name_home_team=key, name_away_team=GAMES_FOR_AVERAGE[key],
                                                             teams_2020=teams_2020, classifier=classifier, list_attr=list_attr,
                                                             clf_index=clf_index, regressor=regressor, list_attr2=list_attr2,
                                                             reg_index=reg_index, get_average=True)

            sum_fitness_max = sum_fitness_max + fitness_by_max_initial_team
            sum_steps_max = sum_steps_max + steps_max
            sum_fitness_rs = sum_fitness_rs + fitness_by_rs_initial
            sum_steps_rs = sum_steps_rs + steps_rs
            sum_fitness_max_adv = sum_fitness_max_adv + fitness_by_max_initial_team_adv
            sum_steps_max_adv = sum_steps_max_adv + steps_max_adv
            sum_fitness_rs_adv = sum_fitness_rs_adv + fitness_by_rs_initial_adv
            sum_steps_rs_adv = sum_steps_rs_adv + steps_rs_adv
            sum_fitness_max_xi = sum_fitness_max_xi + fitness_max_xi

        list_steps = [1, sum_steps_max / len_games, sum_steps_rs / len_games, sum_steps_max_adv / len_games,
                      sum_steps_rs_adv / len_games]
        list_fitness = [fitness_max_xi / len_games, sum_fitness_max / len_games, sum_fitness_rs / len_games,
                        sum_fitness_max_adv / len_games, sum_fitness_rs_adv / len_games]

        if SHOW_GRAPH:
            show_bar_alogrithm_search(list_to_show=list_steps, title='Average Steps per Algorithm', is_fitness=False)
            show_bar_alogrithm_search(list_to_show=list_fitness, title='Average Score per Algorithm', is_fitness=True)
            print(list_fitness)
            print(list_steps)
