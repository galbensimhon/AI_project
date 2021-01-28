from utils.class_and_reg import *
from utils.graphs import *
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.model_selection import KFold

FIRST_N = 1
LAST_N = 3
KNN_CLF_N = 116
KNN_REG_N = 86


# TODO: change func name and train_data/norm_train_data, test_data/norm_test_data in case of testing (line 33-36)
# normalize data - for ADAboost, KNN.
# for all others - normal data

# TODO: change name title and print in case of testing (line 109-113)

def check_classifiers_and_regressors(train_data, train_target_clf, train_target_reg, test_data, test_target,
                                     train_norm_data, test_norm_data, k):
    """
    calculate average accuracy score for specific 'name' classifier prediction,
    and set into names_scores dict.
    :param train_data: the train data
    :param train_target_clf: the train target clf
    :param train_target_reg: the train target reg
    :param test_data: the test data
    :param test_target: the test target
    :param train_norm_data: the train normalize data
    :param test_norm_data: the test normalize data
    :param k: k params for printing
    """

    score_clf = calculate_score(train_data=train_norm_data, train_target=train_target_clf, test_data=test_norm_data,
                                test_target=test_target, func=ada_clf)
    score_reg = calculate_score(train_data=train_norm_data, train_target=train_target_reg, test_data=test_norm_data,
                                test_target=test_target, func=ada_clf_reg)
    print('calculate for ' + str(k) + ', clf score: ' + str(score_clf) + ', reg score: ' + str(score_reg))
    return score_clf, score_reg


def select_k_best_features(train, score_func, k):
    train_data = train[train.columns.values[:-1]]
    train_target = train[train.columns.values[-1]]
    best_features = SelectKBest(score_func=score_func, k=k)
    fit = best_features.fit(train_data, train_target)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(train_data.columns)
    feature_scores = pd.concat([df_columns, df_scores], axis=1)
    feature_scores.columns = ['Specs', 'Score']
    ret_list = list(feature_scores.nlargest(k, 'Score')['Specs'])
    ret_list.append('diff_score')
    return ret_list


if __name__ == '__main__':

    combine = pd.read_csv('databases\combine_databases\combine_game_results.csv', engine='python')
    combine = combine[combine.columns.values[1:]]
    combine = combine[:5306]
    reg_combine = pd.read_csv('databases\combine_databases\combine.csv', engine='python')
    reg_combine = reg_combine[reg_combine.columns.values[1:-1]]
    reg_combine = reg_combine[:5306]
    combine = prepare_label(combine)
    reg_combine = prepare_label(reg_combine)
    dict_list = list(range(1,11))
    classify_dict = dict()
    for name in dict_list:
        classify_dict[name] = {
            'clf': list(),
            'reg': list()
        }
    mekor = combine
    mekor_reg = reg_combine

    for k in dict_list:
        index_clf = select_k_best_features(train=mekor, score_func=f_classif, k=k)
        reg_index = select_k_best_features(train=mekor_reg, score_func=f_regression, k=k)
        combine = mekor[index_clf]
        reg_combine = mekor_reg[reg_index]
        combine_norm_matrix, combine_matrix = prepare_matrix_to_clf(matrix=combine)
        reg_combine_norm_matrix, reg_combine_matrix = prepare_matrix_to_clf(matrix=reg_combine)
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
                score_clf, score_reg = check_classifiers_and_regressors(train_data=X_train, train_target_clf=y_train,
                                                                        train_target_reg=y_reg_train, test_data=X_test,
                                                                        test_target=y_test, train_norm_data=X_norm_train,
                                                                        test_norm_data=X_norm_test, k=k)
                classify_dict[k]['clf'].append(score_clf)
                classify_dict[k]['reg'].append(score_reg)

    clf_res = list()
    reg_res = list()
    for name in dict_list:
        clf_list = classify_dict[name]['clf']
        reg_list = classify_dict[name]['reg']
        clf_res.append(get_avg(clf_list))
        reg_res.append(get_avg(reg_list))
    print("ada clf list :")
    print(clf_res)
    print("ada reg list :")
    print(reg_res)
    show_line_graph(res_clf=clf_res, res_rgs=reg_res, title='K-best-features for Ada boost', dict_list=dict_list)
    k_max_clf = clf_res.index(max(clf_res)) + 1
    k_max_reg = reg_res.index(max(reg_res)) + 1
    print("k clf is " + str(k_max_clf))
    print("k reg is " + str(k_max_reg))
