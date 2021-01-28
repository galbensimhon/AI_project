import pandas as pd
import numpy as np
import sklearn
import requests
import re

from sklearn import preprocessing
from bs4 import BeautifulSoup


# ------------------------- PARAMS -------------------------

CHECK_CLASSIFIERS_AND_REGRESSIONS = 1
PRD_BY_DIFF_SCORE = 0
NUM_LOOPS = 1
CONVERT_TO_TIE = 0.2
GET_AVG_GRAPH = 1
SHOW_VALUES = 1
K_FOLD = 14

KNN_CLF_N = 123
KNN_REG_N = 83
ADA_N_EST_CLF = 5
ADA_N_EST_REG = 5
DT_MAX_CLF = 3
DT_MAX_REG = 2
RF_MAX_DEPTH_CLF = 39
RF_MAX_DEPTH_REG = 15
RF_N_EST_CLF = 501
RF_N_EST_REG = 751

MLP_SOLVER_CLF = 'sgd'
MLP_SOLVER_REG = 'adam'
MLP_ACT_CLF = 'logistic'
MLP_ACT_REG = 'logistic'

SELECT_K_BEST = 1
K_BEST = 40

INT_TO_NAME = {
    1: 'adam+identity',
    2: 'adam+logistic',
    3: 'adam+tanh',
    4: 'adam+relu',
    5: 'lbfgs+identity',
    6: 'lbfgs+logistic',
    7: 'lbfgs+tanh',
    8: 'lbfgs+relu',
    9: 'sgd+identity',
    10: 'sgd+logistic',
    11: 'sgd+tanh',
    12: 'sgd+relu'
}

NATAION_TO_RANKING = {}

TEAM_TO_RANKING = {
    'Everton': 1,
    'Sheffield United': 2,
    'Leicester City': 3,
    'Chelsea': 4,
    'Liverpool': 5,
    'Southampton': 6,
    'Norwich City': 7,
    'Burnley': 8,
    'Manchester United': 9,
    'Crystal Palace': 10,
    'Wolverhampton Wanderers': 11,
    'West Ham United': 12,
    'Tottenham Hotspur': 13,
    'Manchester City': 14,
    'Arsenal': 15,
    'AFC Bournemouth': 16,
    'Brighton & Hove Albion': 17,
    'Watford': 18,
    'Newcastle United': 19,
    'Aston Villa': 20
}
# ------------------------- END PARAMS -------------------------

# ------------------------- HELPERS FUNCTIONS -------------------------


def make_nation_ranking():
    """
    create dictionary of each nation per her rank in world FIFA ranking
    :return:
    """
    if len(NATAION_TO_RANKING) != 0:
        return

    webpage = requests.get('https://www.fifa.com/fifa-world-ranking/ranking-table/men/')
    soup = BeautifulSoup(webpage.text, 'html.parser')
    teams = soup.find_all('tr', attrs={'data-team-id': re.compile('[1-9]*')})
    i = 1

    for team in teams:
        name = team.find(attrs={'class': 'fi-t__nText'}).text
        NATAION_TO_RANKING[name] = i
        i += 1
    # some fixes for 'join' databases
    NATAION_TO_RANKING['United States'] = 26
    NATAION_TO_RANKING['Trinidad & Tobago'] = 104
    NATAION_TO_RANKING['Bosnia & Herzegovina'] = 50
    NATAION_TO_RANKING['DR Congo'] = 57
    NATAION_TO_RANKING['Curacao'] = 80
    NATAION_TO_RANKING['Iran'] = 30
    NATAION_TO_RANKING['Cape Verde'] = 78
    NATAION_TO_RANKING['Antigua & Barbuda'] = 126
    NATAION_TO_RANKING['Ivory Coast'] = 60


def make_dict_list(keys):
    """
    create dict of lists by keys: dict_list
    :param keys: the keys list
    :return: dict of lists
    """
    classify_dict = dict()
    for name in keys:
        classify_dict[name] = {
            'clf': list(),
            'reg': list()
        }
    return classify_dict


def get_avg(list):
    """
    return average of list
    :param list: the given input
    :return: return average of lists
    """
    return sum(list) / len(list)


def get_avg_lists(dict_list, class_dict):
    """
    return list of average score from classify and regression
    :param dict_list: the dictionary which includes the names of list
    :param class_dict: the values of scores
    :return:
    """
    clf_res = list()
    reg_res = list()
    max_clf = 0
    max_reg = 0
    max_name = ''
    max_reg_name = ''

    for name in dict_list:
        clf_list = class_dict[name]['clf']
        reg_list = class_dict[name]['reg']
        avg_clf = get_avg(clf_list)
        avg_reg = get_avg(reg_list)
        if avg_clf >= max_clf:
            max_clf = avg_clf
            max_name = name
        if avg_reg >= max_reg:
            max_reg_name = name
            max_reg = avg_reg
        clf_res.append(avg_clf)
        reg_res.append(avg_reg)

    return clf_res, reg_res, max_name, max_reg_name


def make_trinity_array(prd):
    """
    convert regression result array to trinity (one of: 1, 0, -1)
    :param prd: array to convert
    :return: prediction array after converting
    """
    prd_to_return = list()
    for i in range(len(prd) - 1):
        if prd[i] > CONVERT_TO_TIE:
            prd_to_return.append(1)
        elif prd[i] < -CONVERT_TO_TIE:
            prd_to_return.append(-1)
        else:
            prd_to_return.append(0)
    return prd_to_return


def normalize_vector(matrix, train_matrix):
    """
    normalize matrix with same dimensions by train_matrix by calculation:
    (origin_val - min_train_val) / (max_train_val - min_train_val)
    :param matrix: the matrix which we want to normalize
    :param train_matrix: train matrix which normalize by her

    :return: normalize matrix
    """
    number_of_cols = len(matrix[0])
    matrix_to_return = matrix
    for j in range(number_of_cols - 1):
        if isinstance(train_matrix[0][j], str):
            print("jump on " + train_matrix[0][j])
            continue
        minimum = np.Inf
        maximum = -np.Inf
        for i in range(len(train_matrix)):
            minimum = min([minimum, float(train_matrix[i][j])])
            maximum = max([maximum, float(train_matrix[i][j])])
        for i in range(len(matrix)):
            matrix_to_return[i][j] = (matrix_to_return[i][j] - minimum) / (maximum - minimum)
    return pd.DataFrame(matrix_to_return)


def calc_accuracy_score(clf, train_data, train_target, test_data, test_target, is_clf: bool):
    """
    calculate the accuracy of classifier prediction
    :param clf: the specific classifier / regression
    :param train_data: the train data
    :param train_target: the train target
    :param test_data: the test data
    :param test_target: the test target
    :param is_clf: boolean value: True- calc for classifier, False- calc for regression
    :return: the accuracy of prediction
    """
    clf.fit(train_data, train_target)
    prd = clf.predict(test_data)
    return sklearn.metrics.accuracy_score(test_target, prd) if is_clf else \
        sklearn.metrics.accuracy_score(make_trinity_array(prd=test_target), make_trinity_array(prd=prd))


def calculate_score(train_data, train_target, test_data, test_target, func, n=None, k=None):
    """
    calculate accuracy score for specific func prediction
    :param train_data: the train data
    :param train_target: the train target
    :param test_data: the test data
    :param test_target: the test target
    :param func: specific classifier to predict
    :param n: first param for func
    :param k: second param for func
    :return: accuracy score for specific func prediction
    """
    acc_list = list()
    for i in range(NUM_LOOPS):
        acc_list.append(func(train_data, train_target, test_data, test_target, n=n, k=k))
    return sum(acc_list) / len(acc_list)


def prepare_label(matrix):
    le = preprocessing.LabelEncoder()
    make_nation_ranking()
    # need to make sure make nation to ranking already run
    matrix = matrix.replace(NATAION_TO_RANKING)
    matrix = matrix.replace(TEAM_TO_RANKING)
    for column_name in matrix.columns:
        if matrix[column_name].dtype == object:
            # case we at feature which includes string like "nation" feature
            matrix[column_name] = le.fit_transform(matrix[column_name])
        else:
            pass
    return matrix


def prepare_matrix_to_clf(matrix):
    """
    prepare dataframe to classifier
    :param matrix: the data frame
    :return: normalize dataframe & normal dataframe
    """
    le = preprocessing.LabelEncoder()
    for column_name in matrix.columns:
        if matrix[column_name].dtype == object:
            # case we at feature which includes string like "nation" feature
            matrix[column_name] = le.fit_transform(matrix[column_name])
        else:
            pass
    train_matrix_before_norm = matrix
    return normalize_vector(train_matrix_before_norm.values, train_matrix_before_norm.values), train_matrix_before_norm

# ------------------------- END HELPERS FUNCTIONS -------------------------
