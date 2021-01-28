
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.model_selection import KFold
from utils.class_and_reg import *
from utils.graphs import *

# TODO: GET_AVG_GRAPH = 1, CHECK_CLASSIFIERS_AND_REGRESSIONS = 1


def run_loop_for_scores(train_data, train_target_clf, train_target_reg, test_data, test_target, name: str, func_clf,
                        func_reg, names_scores, index_clf, index_reg):
    """
    calculate average accuracy score for specific 'name' classifier prediction,
    and set into names_scores dict.
    :param train_data: the train data
    :param train_target_clf: the train target
    :param train_target_reg: the train target
    :param test_data: the test data
    :param test_target: the test target
    :param name: classifier name
    :param func_clf: predict by classify
    :param func_reg: predict by regressor
    :param names_scores: dict() - save for each 'name' classifier his scores (score by regressor & score by classifiy)
    :param index_clf: dict() - index of classify
    :param index_reg: dict() - index of regression

    """
    score_clf = calculate_score(train_data=train_data[index_clf[name]], train_target=train_target_clf,
                                test_data=test_data[index_clf[name]], test_target=test_target, func=func_clf)
    score_reg = calculate_score(train_data=train_data[index_reg[name]], train_target=train_target_reg,
                                test_data=test_data[index_reg[name]], test_target=test_target, func=func_reg)
    names_scores[name] = {
        'clf': score_clf,
        'reg': score_reg
    }
    print('calculate for ' + str(name) + ', clf score: ' + str(score_clf) + ', reg score: ' + str(score_reg))


def check_classifiers_and_regressors(train_data, train_target_clf, train_target_reg, test_data, test_target,
                                     train_norm_data, test_norm_data, names_scores, reg_index, clf_index):
    """
    calculate all classifiers by classify and regression, and set in names_scores dict()
    :param train_data: the train data
    :param train_target_clf: the train target
    :param train_target_reg: the train target
    :param test_data: the test data
    :param test_target: the test target
    :param train_norm_data: the train normalize data
    :param test_norm_data: the test normalize data
    :param names_scores: dict() - save for each 'name' classifier his scores (score by regressor & score by classifiy)
    :param clf_index: dict() - index of classify
    :param reg_index: dict() - index of regression
     """
    # calculate accuracy score for data
    for name in ['DT', 'MLP', 'RandForest']:
        run_loop_for_scores(train_data=train_data, train_target_clf=train_target_clf,
                            train_target_reg=train_target_reg, test_data=test_data,
                            test_target=test_target, name=name, func_clf=NAMES_CLF_TO_FUNCTIONS_WITH_PARAMS[name],
                            func_reg=NAMES_REG_TO_FUNCTIONS_WITH_PARAMS[name], names_scores=names_scores,
                            index_clf=clf_index, index_reg=reg_index)

    # calculate accuracy score for normalize data
    for name in ['KNN', 'ADAboost']:
        run_loop_for_scores(train_data=train_norm_data, train_target_clf=train_target_clf,
                            train_target_reg=train_target_reg, test_data=test_norm_data,
                            test_target=test_target, name=name, func_clf=NAMES_CLF_TO_FUNCTIONS_WITH_PARAMS[name],
                            func_reg=NAMES_REG_TO_FUNCTIONS_WITH_PARAMS[name], names_scores=names_scores,
                            index_clf=clf_index, index_reg=reg_index)


# ------------------------- FEATURES SELECTION FUNCTIONS -------------------------

def select_k_best_features(train, score_func, k):
    train_data = train[train.columns.values[:-1]]
    train_target = train[train.columns.values[-1]]
    best_features = SelectKBest(score_func=score_func, k=k)
    fit = best_features.fit(train_data, train_target)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(train_data.columns)
    feature_scores = pd.concat([df_columns, df_scores], axis=1)
    feature_scores.columns = ['Specs', 'Score']  # naming the dataframe columns\
    print(str(k) + "features:")
    print(feature_scores.nlargest(k, 'Score'))
    ret_list = list(feature_scores.nlargest(k, 'Score')['Specs'])
    feature_scores.nlargest(k, 'Score')

    return ret_list


# ------------------------- END FEATURES SELECTION FUNCTIONS -------------------------

if __name__ == '__main__':

    combine = pd.read_csv('databases\combine_databases\combine_game_results.csv', engine='python')
    combine = combine[combine.columns.values[2:]]
    list_attr = combine.columns.values[:-1]
    make_nation_ranking()
    combine = combine[:5306]
    combine = prepare_label(combine)
    clf_index_dict = dict()
    reg_index_dict = dict()
    if SELECT_K_BEST:
        mekor = combine
        clf_index_dict['DT'] = select_k_best_features(train=mekor, score_func=f_classif, k=1)
        clf_index_dict['RandForest'] = select_k_best_features(train=mekor, score_func=f_classif, k=80)
        clf_index_dict['MLP'] = select_k_best_features(train=mekor, score_func=f_classif, k=5)
        clf_index_dict['ADAboost'] = select_k_best_features(train=mekor, score_func=f_classif, k=4)
        clf_index_dict['KNN'] = select_k_best_features(train=mekor, score_func=f_classif, k=60)

    combine_norm_matrix, combine_matrix = prepare_matrix_to_clf(matrix=combine)
    reg_combine = pd.read_csv('databases\combine_databases\combine_diff_score.csv',
                              engine='python')
    reg_combine = reg_combine[reg_combine.columns.values[2:-1]]
    reg_combine = reg_combine[:5306]
    reg_combine = prepare_label(reg_combine)

    if SELECT_K_BEST:
        mekor = reg_combine
        reg_index_dict['DT'] = select_k_best_features(train=mekor, score_func=f_regression, k=80)
        reg_index_dict['RandForest'] = select_k_best_features(train=mekor, score_func=f_regression, k=80)
        reg_index_dict['MLP'] = select_k_best_features(train=mekor, score_func=f_regression, k=10)
        reg_index_dict['ADAboost'] = select_k_best_features(train=mekor, score_func=f_regression, k=100)
        reg_index_dict['KNN'] = select_k_best_features(train=mekor, score_func=f_regression, k=70)

    reg_combine_norm_matrix, reg_combine_matrix = prepare_matrix_to_clf(matrix=reg_combine)

    kf = KFold(n_splits=K_FOLD)
    X = combine_matrix[combine_matrix.columns.values[:-1]].values
    X_norm = combine_norm_matrix[combine_norm_matrix.columns.values[:-1]].values
    Y = combine_matrix[combine_matrix.columns.values[-1]].values
    Z = reg_combine_matrix[reg_combine_matrix.columns.values[-1]].values
    dict_list = {}
    classify_dict = {}
    if GET_AVG_GRAPH:
        dict_list = ['DT', 'MLP', 'RandForest', 'KNN', 'ADAboost']
        classify_dict = make_dict_list(dict_list)

    index = 0

    for train_index, test_index in kf.split(combine_matrix):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        y_reg_train = Z[train_index]
        X_norm_train, X_norm_test = X_norm[train_index], X_norm[test_index]
        names_scores = dict()
        if CHECK_CLASSIFIERS_AND_REGRESSIONS:
            train_x = pd.DataFrame(X_train, columns=list_attr)
            test_x = pd.DataFrame(X_test, columns=list_attr)
            norm_x_tarin = pd.DataFrame(X_norm_train, columns=list_attr)
            norm_x_test = pd.DataFrame(X_norm_test, columns=list_attr)
            check_classifiers_and_regressors(train_data=train_x, train_target_clf=y_train, train_target_reg=y_reg_train,
                                             test_data=test_x, test_target=y_test, train_norm_data=norm_x_tarin,
                                             test_norm_data=norm_x_test, names_scores=names_scores,
                                             reg_index=reg_index_dict, clf_index=clf_index_dict)
            res_clf = list()
            res_rgs = list()
            for element in names_scores:
                res_clf.append(names_scores[element]['clf'])
                res_rgs.append(names_scores[element]['reg'])
            title = 'Classifier VS Regressions ' + '(K-fold: ' + str(index) + ')'

            show_bar_graph(res_clf=res_clf, res_rgs=res_rgs, title=title)

            if GET_AVG_GRAPH:
                for name in dict_list:
                    for clf_reg in ['clf', 'reg']:
                        classify_dict[name][clf_reg].append(names_scores[name][clf_reg])
        index += 1

    if GET_AVG_GRAPH:
        clf_res = list()
        reg_res = list()
        for name in dict_list:
            clf_list = classify_dict[name]['clf']
            reg_list = classify_dict[name]['reg']
            clf_res.append(get_avg(clf_list))
            reg_res.append(get_avg(reg_list))
        show_bar_graph(res_clf=clf_res, res_rgs=reg_res, title='Classifier VS Regressions (avg)')
