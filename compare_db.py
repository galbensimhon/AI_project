from utils.class_and_reg import *
from utils.graphs import *
from sklearn.feature_selection import SelectKBest, f_regression, f_classif


def select_k_best_features(train, score_func, k):
    train_data = train[train.columns.values[:-1]]
    train_target = train[train.columns.values[-1]]
    best_features = SelectKBest(score_func=score_func, k=k)
    fit = best_features.fit(train_data, train_target)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(train_data.columns)
    feature_scores = pd.concat([df_columns, df_scores], axis=1)
    feature_scores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print(str(k) + "features:")
    print(feature_scores.nlargest(k, 'Score'))
    ret_list = list(feature_scores.nlargest(k, 'Score')['Specs'])
    feature_scores.nlargest(k, 'Score')
    return ret_list


if __name__ == '__main__':
    combine = pd.read_csv('databases/combine_databases/combine_game_results.csv', engine='python')
    combine = combine[combine.columns.values[2:]]
    list_attr = combine.columns.values[:-1]
    combine = prepare_label(combine)
    combine_test = combine[5306:]
    combine = combine[:5306]
    clf_index = select_k_best_features(train=combine, score_func=f_classif, k=80)
    reg_combine = pd.read_csv('databases/combine_databases/combine_diff_score.csv', engine='python')
    reg_combine = reg_combine[reg_combine.columns.values[2:-1]]
    reg_combine = prepare_label(reg_combine)
    reg_test = reg_combine[5306:]
    reg_combine = reg_combine[:5306]
    reg_index = select_k_best_features(train=reg_combine, score_func=f_regression, k=80)

    avg_combine = pd.read_csv('databases/combine_databases/combine_avg_game_results.csv', engine='python')
    avg_combine = avg_combine[avg_combine.columns.values[2:]]
    avg_combine = prepare_label(avg_combine)
    clf_index_avg = select_k_best_features(train=avg_combine, score_func=f_classif, k=80)
    avg_combine_test = avg_combine[5306:]
    avg_combine = avg_combine[:5306]

    avg_reg_combine = pd.read_csv('databases/combine_databases/combine_avg_diff_score.csv', engine='python')
    avg_reg_combine = avg_reg_combine[avg_reg_combine.columns.values[2:-1]]
    avg_reg_combine = prepare_label(avg_reg_combine)
    reg_index_avg = select_k_best_features(train=avg_reg_combine, score_func=f_regression, k=80)
    avg_reg_combine_test = avg_reg_combine[5306:]
    avg_reg_combine = avg_reg_combine[:5306]

    delete_combine = pd.read_csv('databases/combine_databases/combine_with_delete_game_results.csv', engine='python')
    delete_combine = delete_combine[delete_combine.columns.values[2:]]
    delete_combine = prepare_label(delete_combine)
    clf_index_delete = select_k_best_features(train=delete_combine, score_func=f_classif, k=80)
    delete_combine_test = delete_combine[3124:]
    delete_combine = delete_combine[:3124]

    delete_reg_combine = pd.read_csv('databases/combine_databases/combine_with_delete_diff_score.csv', engine='python')
    delete_reg_combine = delete_reg_combine[delete_reg_combine.columns.values[2:-1]]
    delete_reg_combine = prepare_label(delete_reg_combine)
    reg_index_del = select_k_best_features(train=delete_reg_combine, score_func=f_regression, k=80)
    delete_reg_combine_test = delete_reg_combine[3124:]
    delete_reg_combine = delete_reg_combine[:3124]

    X = combine[combine.columns.values[:-1]]
    Y = combine[combine.columns.values[-1]].values
    X_test = combine_test[combine_test.columns.values[:-1]]
    Y_test = combine_test[combine_test.columns.values[-1]].values

    Y_reg = reg_combine[reg_combine.columns.values[-1]].values
    Y_reg_test = reg_test[reg_test.columns.values[-1]].values

    list_clf = list()
    list_reg = list()

    list_clf.append(random_forest_clf(train_data=X[clf_index], train_target=Y, test_data=X_test[clf_index],
                                      test_target=Y_test))
    list_reg.append(random_forest_clf_reg(train_data=X[reg_index], train_target=Y_reg, test_data=X_test[reg_index],
                                          test_target=Y_reg_test))

    X = avg_combine[avg_combine.columns.values[:-1]]
    Y = avg_combine[avg_combine.columns.values[-1]].values
    X_test = avg_combine_test[avg_combine_test.columns.values[:-1]]
    Y_test = avg_combine_test[avg_combine_test.columns.values[-1]].values

    Y_reg = avg_reg_combine[avg_reg_combine.columns.values[-1]].values
    Y_reg_test = avg_reg_combine_test[avg_reg_combine_test.columns.values[-1]].values


    list_clf.append(random_forest_clf(train_data=X[clf_index_avg], train_target=Y, test_data=X_test[clf_index_avg],
                                      test_target=Y_test))
    list_reg.append(random_forest_clf_reg(train_data=X[reg_index_avg], train_target=Y_reg,
                                          test_data=X_test[reg_index_avg], test_target=Y_reg_test))

    X = delete_combine[delete_combine.columns.values[:-1]]
    Y = delete_combine[delete_combine.columns.values[-1]].values
    X_test = delete_combine_test[delete_combine_test.columns.values[:-1]]
    Y_test = delete_combine_test[delete_combine_test.columns.values[-1]].values

    Y_reg = delete_reg_combine[delete_reg_combine.columns.values[-1]].values
    Y_reg_test = delete_reg_combine_test[delete_reg_combine_test.columns.values[-1]].values
    list_clf.append(random_forest_clf(train_data=X[clf_index_delete], train_target=Y, test_data=X_test[clf_index_delete],
                                      test_target=Y_test))
    list_reg.append(random_forest_clf_reg(train_data=X[reg_index_del], train_target=Y_reg,
                                          test_data=X_test[reg_index_del], test_target=Y_reg_test))

    print(list_clf)
    print(list_reg)
    show_bar_graph_for_DB(res_clf=list_clf,res_rgs=list_reg, title="Compare DataBase")



