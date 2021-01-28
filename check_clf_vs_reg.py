from utils.class_and_reg import *
from utils.graphs import *
from sklearn.model_selection import KFold

# TODO:
# 1. change FIRST_N, LAST_N, STEP  for specific test:
# DT: 1-20-1, RF: 1-20-2(depth) RF_EST: 1-1001-50(estimator), MLP: 1-13-1, ADA: 1-81-1, KNN: 1-200-1
# 2. change K_FOLD if necessary
# 3. change KEY_TO_CHECK to specific test

FIRST_N = 1
LAST_N = 5
STEP = 2

GRAPH_PER_K_FOLD = 0

MLP = 'mlp'
DT = 'decision_tree'
RF = 'random_forest'
RF_EST = 'random_forest_est'
KNN = 'knn'
ADA = 'adaboost'

KEY_TO_CHECK = DT

# ------------------------- HELPERS FUNCTIONS -------------------------


def initial():
    """
    create initial matrix about processing the databases
    :return: matrixes and dictionaries to work with
    """
    combine = pd.read_csv('databases\combine_databases\combine_game_results.csv', engine='python')
    combine = combine[combine.columns.values[2:]]
    combine = combine[:5306]
    combine_norm_matrix, combine_matrix = prepare_matrix_to_clf(matrix=combine)

    reg_combine = pd.read_csv('databases\combine_databases\combine_diff_score.csv', engine='python')
    reg_combine = reg_combine[reg_combine.columns.values[2:-1]]
    reg_combine = reg_combine[:5306]
    reg_combine_norm_matrix, reg_combine_matrix = prepare_matrix_to_clf(matrix=reg_combine)

    dict_list = list(range(FIRST_N, LAST_N, STEP))
    classify_dict = dict()
    for name in dict_list:
        classify_dict[name] = {
            'clf': list(),
            'reg': list()
        }
    return combine_matrix, combine_norm_matrix, reg_combine_matrix, reg_combine_norm_matrix, dict_list, classify_dict

# ------------------------- END HELPERS FUNCTIONS -------------------------


def check_classifiers_and_regressors(train_data, train_norm_data, train_target_clf, train_target_reg, test_data,
                                     test_norm_data, test_target, names_scores, key):
    """
    calculate all classifiers by classify and regression, and set in names_scores dict()
    :param train_data: the train data
    :param train_norm_data: the train normalize data
    :param train_target_clf: the train target
    :param train_target_reg: the train target
    :param test_data: the test data
    :param test_norm_data: the test normalize data
    :param test_target: the test target
    :param train_norm_data: the train normalize data
    :param test_norm_data: the test normalize data
    :param names_scores: dict() - save for each 'name' classifier his scores (score by regressor & score by classifiy)
    :param key: one of: (MLP, DT, ADA, RF, KNN)
     """
    # calculate accuracy score for data
    if key == MLP:
        calculate_mlp(train_data=train_data, test_data=test_data, test_target=test_target,
                      train_target_clf=train_target_clf, train_target_reg=train_target_reg, names_scores=names_scores)
        for name in dict_list:
            for clf_reg in ['clf', 'reg']:
                classify_dict[name][clf_reg].append(names_scores[INT_TO_NAME[name]][clf_reg])
        return classify_dict
    elif key == DT:
        calculate_dt(train_data=train_data, test_data=test_data, test_target=test_target,
                     train_target_clf=train_target_clf, train_target_reg=train_target_reg, names_scores=names_scores)
        for name in dict_list:
            for clf_reg in ['clf', 'reg']:
                classify_dict[name][clf_reg].append(names_scores[name][clf_reg])
        return classify_dict
    elif key == ADA:
        calculate_ada(train_norm_data=train_norm_data, test_norm_data=test_norm_data, test_target=test_target,
                      train_target_clf=train_target_clf, train_target_reg=train_target_reg, names_scores=names_scores)
        for name in dict_list:
            for clf_reg in ['clf', 'reg']:
                classify_dict[name][clf_reg].append(names_scores[name][clf_reg])
        return classify_dict
    elif key == RF:
        calculate_rf_depth(train_data=train_data, test_data=test_data, test_target=test_target,
                           train_target_clf=train_target_clf, train_target_reg=train_target_reg,
                           names_scores=names_scores)
        for name in dict_list:
            for clf_reg in ['clf', 'reg']:
                classify_dict[name][clf_reg].append(names_scores[name][clf_reg])
        return classify_dict
    elif key == RF_EST:
        calculate_rf_est(train_data=train_data, test_data=test_data, test_target=test_target,
                         train_target_clf=train_target_clf, train_target_reg=train_target_reg,
                         names_scores=names_scores)
        for name in dict_list:
            for clf_reg in ['clf', 'reg']:
                classify_dict[name][clf_reg].append(names_scores[name][clf_reg])
        return classify_dict
    elif key == KNN:
        calculate_knn(train_data=train_norm_data, test_data=test_norm_data, test_target=test_target,
                      train_target_clf=train_target_clf, train_target_reg=train_target_reg,
                      names_scores=names_scores)
        for name in dict_list:
            for clf_reg in ['clf', 'reg']:
                classify_dict[name][clf_reg].append(names_scores[name][clf_reg])
        return classify_dict


def calculate_mlp(train_data, test_data, test_target, train_target_clf, train_target_reg, names_scores):
    for solve in ['adam', 'lbfgs', 'sgd']:
        for active in ['identity', 'logistic', 'tanh', 'relu']:
            score_clf = calculate_score(train_data=train_data, train_target=train_target_clf,
                                        test_data=test_data, test_target=test_target, func=mlp_clf, n=solve, k=active)
            score_reg = calculate_score(train_data=train_data, train_target=train_target_reg,
                                        test_data=test_data, test_target=test_target, func=mlp_clf_reg, n=solve,
                                        k=active)
            names_scores[solve + '+' + active] = {
                'clf': score_clf,
                'reg': score_reg
            }
            print('calculate for ' + str(solve+'+'+active) + ', clf score: ' + str(score_clf) + ', reg score: ' +
                  str(score_reg))


def calculate_dt(train_data, test_data, test_target, train_target_clf, train_target_reg, names_scores):
    for n in range(FIRST_N, LAST_N, STEP):
        score_clf = calculate_score(train_data=train_data, train_target=train_target_clf,
                                    test_data=test_data, test_target=test_target, func=decision_tree_clf, n=n)
        score_reg = calculate_score(train_data=train_data, train_target=train_target_reg,
                                    test_data=test_data, test_target=test_target, func=decision_tree_clf_reg, n=n)
        names_scores[n] = {
            'clf': score_clf,
            'reg': score_reg
        }
        print('calculate for ' + str(n) + ', clf score: ' + str(score_clf) + ', reg score: ' + str(score_reg))


def calculate_rf_depth(train_data, test_data, test_target, train_target_clf, train_target_reg, names_scores):
    for n in range(FIRST_N, LAST_N, STEP):
        score_clf = calculate_score(train_data=train_data, train_target=train_target_clf,
                                    test_data=test_data, test_target=test_target, func=random_forest_clf, n=n, k=100)
        score_reg = calculate_score(train_data=train_data, train_target=train_target_reg,
                                    test_data=test_data, test_target=test_target, func=random_forest_clf_reg, n=n,
                                    k=100)
        names_scores[n] = {
            'clf': score_clf,
            'reg': score_reg
        }
        print('calculate for ' + str(n) + ', clf score: ' + str(score_clf) + ', reg score: ' + str(score_reg))


def calculate_rf_est(train_data, test_data, test_target, train_target_clf, train_target_reg, names_scores):
    for n in range(FIRST_N, LAST_N, STEP):
        score_clf = calculate_score(train_data=train_data, train_target=train_target_clf,
                                    test_data=test_data, test_target=test_target, func=random_forest_clf, n=39, k=n)
        score_reg = calculate_score(train_data=train_data, train_target=train_target_reg,
                                    test_data=test_data, test_target=test_target, func=random_forest_clf_reg, n=13, k=n)
        names_scores[n] = {
            'clf': score_clf,
            'reg': score_reg
        }
        print('calculate for ' + str(n) + ', clf score: ' + str(score_clf) + ', reg score: ' + str(score_reg))


def calculate_ada(train_norm_data, test_norm_data, test_target, train_target_clf, train_target_reg, names_scores):
    for n in range(FIRST_N, LAST_N, STEP):
        score_clf = calculate_score(train_data=train_norm_data, train_target=train_target_clf,
                                    test_data=test_norm_data, test_target=test_target, func=ada_clf, n=n)
        score_reg = calculate_score(train_data=train_norm_data, train_target=train_target_reg,
                                    test_data=test_norm_data, test_target=test_target, func=ada_clf_reg, n=n)
        names_scores[n] = {
            'clf': score_clf,
            'reg': score_reg
        }
        print('calculate for ' + str(n) + ', clf score: ' + str(score_clf) + ', reg score: ' + str(score_reg))


def calculate_knn(train_data, test_data, test_target, train_target_clf, train_target_reg, names_scores):
    for n in range(FIRST_N, LAST_N, STEP):
        score_clf = calculate_score(train_data=train_data, train_target=train_target_clf,
                                    test_data=test_data, test_target=test_target, func=knn_clf, n=n)
        score_reg = calculate_score(train_data=train_data, train_target=train_target_reg,
                                    test_data=test_data, test_target=test_target, func=knn_clf_reg, n=n)
        names_scores[n] = {
            'clf': score_clf,
            'reg': score_reg
        }
        print('calculate for ' + str(n) + ', clf score: ' + str(score_clf) + ', reg score: ' + str(score_reg))


if __name__ == '__main__':

    combine_matrix, combine_norm_matrix, reg_combine_matrix, reg_combine_norm_matrix, dict_list, classify_dict = \
        initial()

    kf = KFold(n_splits=K_FOLD)
    X = combine_matrix[combine_matrix.columns.values[:-1]].values
    X_norm = combine_norm_matrix[combine_norm_matrix.columns.values[:-1]].values
    Y = combine_matrix[combine_matrix.columns.values[-1]].values
    Z = reg_combine_matrix[reg_combine_matrix.columns.values[-1]].values

    index = 0
    for train_index, test_index in kf.split(combine_matrix):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        y_reg_train = Z[train_index]
        X_norm_train, X_norm_test = X_norm[train_index], X_norm[test_index]
        names_scores = dict()
        if CHECK_CLASSIFIERS_AND_REGRESSIONS:
            classify_dict = check_classifiers_and_regressors(train_data=X_train, train_norm_data=X_norm_train,
                                                             train_target_clf=y_train, train_target_reg=y_reg_train,
                                                             test_data=X_test, test_norm_data=X_norm_test,
                                                             test_target=y_test, names_scores=names_scores,
                                                             key=KEY_TO_CHECK)
        index += 1

        # relevant only for checking KNN
        if GRAPH_PER_K_FOLD:
            res_clf = list()
            res_rgs = list()
            for element in names_scores:
                res_clf.append(names_scores[element]['clf'])
                res_rgs.append(names_scores[element]['reg'])
            title = 'KNN find best n' + '(K-fold: ' + str(index) + ')'

    clf_res, reg_res, max_name, max_reg_name = get_avg_lists(dict_list=dict_list, class_dict=classify_dict)

    # FOR MLP:
    if KEY_TO_CHECK == MLP:
        print('max DT for clf: ' + str(max_name))
        print('max DT for reg: ' + str(max_reg_name))
        show_bar_graph_mlp(res_clf=clf_res, res_rgs=reg_res)
    # FOR Decision-Tree:
    elif KEY_TO_CHECK == DT:
        print('max DT for clf: ' + str(max_name))
        print('max DT for reg: ' + str(max_reg_name))
        show_line_graph_dt(res_clf=clf_res, res_rgs=reg_res, first=FIRST_N, last=LAST_N, step=STEP)
    # FOR Random-Forest(depth):
    elif KEY_TO_CHECK == RF:
        print('max RF for clf: ' + str(max_name))
        print('max RF for reg: ' + str(max_reg_name))
        show_line_graph_rf(res_clf=clf_res, res_rgs=reg_res, first=FIRST_N, last=LAST_N, step=STEP)
    # FOR ADAboost:
    elif KEY_TO_CHECK == ADA:
        print('max ADA for clf: ' + str(max_name))
        print('max ADA for reg: ' + str(max_reg_name))
        show_line_graph_ada(res_clf=clf_res, res_rgs=reg_res, first=FIRST_N, last=LAST_N, step=STEP)
    # FOR RandomForest(esti):
    elif KEY_TO_CHECK == RF_EST:
        print('max RF for clf: ' + str(max_name))
        print('max RF for reg: ' + str(max_reg_name))
        show_line_graph_rf_est(res_clf=clf_res, res_rgs=reg_res, first=FIRST_N, last=LAST_N, step=STEP)
    # FOR KNN:
    elif KEY_TO_CHECK == KNN:
        print('max KNN for clf: ' + str(max_name))
        print('max KNN for reg: ' + str(max_reg_name))
        show_line_graph_knn(res_clf=clf_res, res_rgs=reg_res, first=FIRST_N, last=LAST_N, step=STEP)
