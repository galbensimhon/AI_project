import sys

from utils.utils_search import *
from utils.graphs import *

# TODO: should change
NUM_OF_ITER = 1
USE_ADVANCED_LOCAL_SEARCH_ITER = 1
MAX_SEARCH_STEPS_ITER = 5


def swap_teams(team1: Team, team2: Team):
    return team2, team1


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


def get_score_by_heuristic(home_xi_list: list, away_xi_list: list, reg, clf, clf_index, reg_index, list_attr,
                           list_attr2, home_name, away_name):
    """
    This function calculate the improvement in each iteration of new lineup.
    for example: diff between the first and second lineup of home team against the lineup of away team

    :param home_xi_list: list of lineups of home_team which improved while running
    :param away_xi_list: list of lineups of away_team which improved while running
    :param reg: the regressor
    :param clf: the classify
    :param clf_index: index for classify
    :param reg_index: index of regressor
    :param list_attr: attributes for classify
    :param list_attr2: attributes for regressor
    :param home_name: home name
    :param away_name: away name

    :return: list of all differents between lineup imporvent.
    in addition calculate the difference between first lineup for home team, and the last lineup for home team.
    """

    diff_list = list()
    for i in range(1, len(home_xi_list)):
        away_list = xi_to_list(team_name=away_name, list_players=away_xi_list[i-1])
        heuristic_by_second_lineup = heuristic(reg=reg, clf=clf, tmp_xi=home_xi_list[i], clf_index=clf_index,
                                               reg_index=reg_index, name=home_name, list_attr=list_attr,
                                               list_attr2=list_attr2, away_list=away_list)
        heuristic_by_first_lineup = heuristic(reg=reg, clf=clf, tmp_xi=home_xi_list[i-1], clf_index=clf_index,
                                              reg_index=reg_index, name=home_name, list_attr=list_attr,
                                              list_attr2=list_attr2, away_list=away_list)
        diff_list.append(heuristic_by_second_lineup - heuristic_by_first_lineup)

    away_list = xi_to_list(team_name=away_name, list_players=away_xi_list[-1])
    heuristic_by_last_lineup = heuristic(reg=reg, clf=clf, tmp_xi=home_xi_list[-1], clf_index=clf_index,
                                         reg_index=reg_index, name=home_name, list_attr=list_attr,
                                         list_attr2=list_attr2, away_list=away_list)
    heuristic_by_first_lineup = heuristic(reg=reg, clf=clf, tmp_xi=home_xi_list[0], clf_index=clf_index,
                                          reg_index=reg_index, name=home_name, list_attr=list_attr,
                                          list_attr2=list_attr2, away_list=away_list)
    diff_score = heuristic_by_last_lineup - heuristic_by_first_lineup

    return diff_list, diff_score, home_xi_list[-1]


def show_graph_improvements(list_values: list, title: str, y_axis=None, x_axis=None):
    """
    create bar chart with values: list_values
    :param list_values: list includes the values
    :param title: the title of graph
    :param y_axis: the name axis of y
    :param x_axis: the name axis of x
    """
    fig, ax = plt.subplots()
    x = list()
    for i in range(1, len(list_values) + 1):
        if title == "Difference-average between two adjacent iterates":
            element = str(i) + '-' + str(i+1)
        else:
            plt.ylim(0, 1.2)
            element = (str(i))
        x.append(element)
    plt.title(title)
    plt.tight_layout()
    width = 0.25
    opacity = 0.8
    react = plt.bar(x, list_values, width, alpha=opacity, color='b')
    plt.xlabel('Iterates' if x_axis is None else x_axis)
    plt.ylabel('Heuristic Score' if y_axis is None else y_axis)
    show_values(ax=ax, rects_clf=react, rects_rgs=[])
    # plt.legend()
    plt.show()


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


def local_search_advanced(team_home, name_home_team, away_list, classifier, list_attr, clf_index, regressor, list_attr2,
                          reg_index, home_xi):
    """
        local search algorithm which check all players that can change specific player
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
        if i >= MAX_SEARCH_STEPS_ITER:
            break
    return cur_max, list_steps, home_xi, m


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
        if i >= MAX_SEARCH_STEPS_ITER:
            break
    return cur_max, list_steps, home_xi, m


def random_restart_search_iter(team_home, name_home_team, away_list, classifier, list_attr, clf_index, regressor, list_attr2,
                               reg_index, home_rep: bool):
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
    :param home_rep: bool, True - in case the lineup its for home team, else False

    :return: lists of steps and fitness for each algorithm
    """
    # list_all = list()
    # list_fitness = list()
    j = 0
    max_all = -np.inf
    max_xi = []
    search_func = local_search_advanced if USE_ADVANCED_LOCAL_SEARCH_ITER else local_search
    while j < NUM_OF_STEP:
        home_xi = make_xi(team_home, (j == 0))
        fitness, list_step, ret_xi, num_step = search_func(team_home=team_home, name_home_team=name_home_team,
                                                           away_list=away_list, classifier=classifier,
                                                           clf_index=clf_index, regressor=regressor,
                                                           reg_index=reg_index, list_attr=list_attr,
                                                           list_attr2=list_attr2, home_xi=home_xi)
        if fitness > max_all:
            max_all = fitness
            max_xi = ret_xi

        # list_all.append(list_step)
        # list_fitness.append(fitness)
        j = j + 1

    team = 'home team:' if home_rep else 'away team:'
    print("The best lineup for " + team)
    print(xi_to_players_names(list_players=max_xi))
    return max_all, max_xi


def search_iter(name_home_team, name_away_team, teams_2020, classifier, list_attr, clf_index, regressor, list_attr2,
                reg_index):
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

    :return: attributes in case of average
    """
    team_home, team_away = prepare_teams(name_home_team=name_home_team, name_away_team=name_away_team,
                                         teams=teams_2020)
    home_name = team_home.name
    away_name = team_away.name
    away_xi = make_xi(team_away, True)
    home_xi_list = list()
    away_xi_list = list()
    home_fitness = 0
    home_rep = True
    for i in range(NUM_OF_ITER):
        print("start iterate: " + str(i))
        away_list = xi_to_list(team_name=team_away.name, list_players=away_xi)
        fitness, away_xi = random_restart_search_iter(team_home=team_home, name_home_team=team_home.name,
                                                      away_list=away_list, classifier=classifier,
                                                      clf_index=clf_index, regressor=regressor,
                                                      reg_index=reg_index, list_attr=list_attr,
                                                      list_attr2=list_attr2, home_rep=home_rep)
        team_home, team_away = swap_teams(team_home, team_away)

        if home_rep:
            home_xi_list.append(away_xi)
            home_fitness = fitness
            home_rep = False
        else:
            away_xi_list.append(away_xi)
            home_rep = True
        print("end iterate: " + str(i))

    diff_list, diff_score, best_xi = \
        get_score_by_heuristic(home_xi_list=home_xi_list, away_xi_list=away_xi_list, reg=regressor,
                               clf=classifier, clf_index=clf_index, reg_index=reg_index, list_attr=list_attr,
                               list_attr2=list_attr2, home_name=home_name, away_name=away_name)
    return diff_list, diff_score, best_xi, home_fitness


def set_list(old_list: list, temp_list: list, count: int):
    """
    add to old_list the values of temp list
    :param old_list: the list to add
    :param temp_list: the values to add
    :param count: numbers of list add
    :return: new list and his count
    """
    if count == 0:
        return temp_list, 1
    else:
        for i in range(len(temp_list)):
            old_list[i] += temp_list[i]
        count += 1
        return old_list, count


if __name__ == '__main__':
    """
    the user can select specific game by home team and away team as input,
    or get average of 20 random games
    """
    classifier, list_attr, clf_index = init_clf()
    regressor, list_attr2, reg_index = init_reg()
    teams_2020 = pd.read_csv('databases/raw_databases/team_2020_lineups.csv', engine='python')
    num_of_params = len(sys.argv)
    name_home_team = str(sys.argv[1]) if num_of_params >= 2 else None
    name_away_team = str(sys.argv[2]) if num_of_params >= 3 else None

    if name_home_team and name_away_team:
        diff_list, diff_score, best_xi, home_fitness = \
            search_iter(name_home_team=name_home_team, name_away_team=name_away_team,
                        teams_2020=teams_2020, classifier=classifier, list_attr=list_attr,
                        clf_index=clf_index, regressor=regressor, list_attr2=list_attr2,
                        reg_index=reg_index)
        print("the improved score between Random-Advanced to Minimax is: " + str(diff_score))
        print("The best lineup for " + name_home_team + ' is: ')
        print(xi_to_players_names(best_xi))
        exit(0)
    elif name_away_team or name_home_team:
        print('Please select 2 names of teams, or none of them')
        exit(-1)
    else:
        list_diff_score = list()
        list_iterate_average = list()
        list_diff_average = list()
        list_fitness_average = list()
        count_iterate = count_diff = 0
        create_dict_games(teams=teams_2020)
        len_games = len(GAMES_FOR_AVERAGE.keys())
        for key in GAMES_FOR_AVERAGE.keys():
            print('--------------------new game start---------------------')
            diff_list, diff_score, best_xi, home_fitness = \
                search_iter(name_home_team=key, name_away_team=GAMES_FOR_AVERAGE[key],
                            teams_2020=teams_2020, classifier=classifier, list_attr=list_attr,
                            clf_index=clf_index, regressor=regressor, list_attr2=list_attr2,
                            reg_index=reg_index)
            list_diff_score.append(diff_score)
            list_fitness_average.append(home_fitness)
            list_diff_average, count_diff = set_list(old_list=list_diff_average, temp_list=diff_list,
                                                     count=count_diff)
        list_diff_average = [i/count_diff for i in list_diff_average]

        if SHOW_GRAPH:
            show_graph_improvements(list_values=list_diff_average, title="Difference-average between two adjacent "
                                                                         "iterates", y_axis='Diff Score')

        print(list_diff_score)
        if SHOW_GRAPH:
            show_graph_improvements(list_values=list_diff_score, title="Difference-average between final lineup and "
                                                                       "first lineup", y_axis='Diff Score', x_axis='Games')

        print("The average diff score between Random Advanced to Minimax is:" + str(get_avg(list_diff_score)))
        print("The average fitness for Minmax: " + str(get_avg(list_fitness_average)))
        exit(0)
