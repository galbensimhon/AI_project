from .utils import *
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ------------------------- CLASSIFIERS FUNCTIONS -------------------------


def mlp_clf(train_data, train_target, test_data, test_target, n=None, k=None):
    clf = MLPClassifier(max_iter=300) if n is None and k is None else MLPClassifier(max_iter=300, solver=n,
                                                                                    activation=k)
    return calc_accuracy_score(clf=clf, train_data=train_data, train_target=train_target, test_data=test_data,
                               test_target=test_target, is_clf=True)


def ada_clf(train_data, train_target, test_data, test_target, n=None, k=None):
    clf = AdaBoostClassifier() if n is None and k is None else AdaBoostClassifier(n_estimators=n)
    return calc_accuracy_score(clf=clf, train_data=train_data, train_target=train_target, test_data=test_data,
                               test_target=test_target, is_clf=True)


def knn_clf(train_data, train_target, test_data, test_target, n=None, k=None):
    clf = KNeighborsClassifier() if n is None and k is None else KNeighborsClassifier(n_neighbors=n)
    return calc_accuracy_score(clf=clf, train_data=train_data, train_target=train_target, test_data=test_data,
                               test_target=test_target, is_clf=True)


def decision_tree_clf(train_data, train_target, test_data, test_target, n=None, k=None):
    clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=2) if n is None and k is None else \
        DecisionTreeClassifier(criterion='entropy', min_samples_split=2, max_depth=n)
    return calc_accuracy_score(clf=clf, train_data=train_data, train_target=train_target, test_data=test_data,
                               test_target=test_target, is_clf=True)


def random_forest_clf(train_data, train_target, test_data, test_target, n=None, k=None):
    clf = RandomForestClassifier(n_estimators=100) if n is None and k is None else \
        RandomForestClassifier(n_estimators=k, bootstrap=True, max_depth=n)
    return calc_accuracy_score(clf=clf, train_data=train_data, train_target=train_target, test_data=test_data,
                               test_target=test_target, is_clf=True)


NAMES_CLF_TO_FUNCTIONS = {
    'DT': decision_tree_clf,
    'KNN': knn_clf,
    'MLP': mlp_clf,
    'ADAboost': ada_clf,
    'RandForest': random_forest_clf
}

# ------------------------- END CLASSIFIERS FUNCTIONS -------------------------

# ------------------------- REGRESSIONS FUNCTIONS -------------------------


def mlp_clf_reg(train_data, train_target, test_data, test_target, n=None, k=None):
    clf = MLPRegressor() if n is None and k is None else MLPRegressor(max_iter=300, solver=n, activation=k)
    return calc_accuracy_score(clf=clf, train_data=train_data, train_target=train_target, test_data=test_data,
                               test_target=test_target, is_clf=False)


def decision_tree_clf_reg(train_data, train_target, test_data, test_target, n=None, k=None):
    clf = DecisionTreeRegressor(min_samples_split=2) if n is None and k is None else \
        DecisionTreeRegressor(min_samples_split=2, max_depth=n)
    return calc_accuracy_score(clf=clf, train_data=train_data, train_target=train_target, test_data=test_data,
                               test_target=test_target, is_clf=False)


def ada_clf_reg(train_data, train_target, test_data, test_target, n=None, k=None):
    clf = AdaBoostRegressor() if n is None and k is None else AdaBoostRegressor(n_estimators=n)
    return calc_accuracy_score(clf=clf, train_data=train_data, train_target=train_target, test_data=test_data,
                               test_target=test_target, is_clf=False)


def random_forest_clf_reg(train_data, train_target, test_data, test_target, n=None, k=None):
    clf = RandomForestRegressor(n_estimators=100) if n is None and k is None else \
        RandomForestRegressor(n_estimators=k, bootstrap=True, max_depth=n)
    return calc_accuracy_score(clf=clf, train_data=train_data, train_target=train_target, test_data=test_data,
                               test_target=test_target, is_clf=False)


def knn_clf_reg(train_data, train_target, test_data, test_target, n=None, k=None):
    clf = KNeighborsRegressor() if n is None and k is None else KNeighborsRegressor(n_neighbors=n)
    return calc_accuracy_score(clf=clf, train_data=train_data, train_target=train_target, test_data=test_data,
                               test_target=test_target, is_clf=False)


NAMES_REG_TO_FUNCTIONS = {
    'DT': decision_tree_clf_reg,
    'KNN': knn_clf_reg,
    'MLP': mlp_clf_reg,
    'ADAboost': ada_clf_reg,
    'RandForest': random_forest_clf_reg
}

# ------------------------- END REGRESSIONS FUNCTIONS -------------------------

# -------------- CLASSIFIERS AND REGRESSIONS FUNCTIONS WITH PARAMS (for compare_all.py) --------------


def decision_tree_clf_param(train_data, train_target, test_data, test_target, n=None, k=None):
    clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=2, max_depth=DT_MAX_CLF)
    return calc_accuracy_score(clf=clf, train_data=train_data, train_target=train_target, test_data=test_data,
                               test_target=test_target, is_clf=True)


def random_forest_clf_param(train_data, train_target, test_data, test_target, n=None, k=None):
    clf = RandomForestClassifier(n_estimators=RF_N_EST_CLF, max_depth=RF_MAX_DEPTH_CLF)
    return calc_accuracy_score(clf=clf, train_data=train_data, train_target=train_target, test_data=test_data,
                               test_target=test_target, is_clf=True)


def knn_clf_param(train_data, train_target, test_data, test_target, n=None, k=None):
    clf = KNeighborsClassifier(n_neighbors=KNN_CLF_N)
    return calc_accuracy_score(clf=clf, train_data=train_data, train_target=train_target, test_data=test_data,
                               test_target=test_target, is_clf=True)


def mlp_clf_param(train_data, train_target, test_data, test_target, n=None, k=None):
    clf = MLPClassifier(activation=MLP_ACT_CLF, max_iter=300, solver=MLP_SOLVER_CLF)
    return calc_accuracy_score(clf=clf, train_data=train_data, train_target=train_target, test_data=test_data,
                               test_target=test_target, is_clf=True)


def ada_clf_param(train_data, train_target, test_data, test_target, n=None, k=None):
    clf = AdaBoostClassifier(n_estimators=ADA_N_EST_CLF)
    return calc_accuracy_score(clf=clf, train_data=train_data, train_target=train_target, test_data=test_data,
                               test_target=test_target, is_clf=True)


NAMES_CLF_TO_FUNCTIONS_WITH_PARAMS = {
    'DT': decision_tree_clf_param,
    'KNN': knn_clf_param,
    'MLP': mlp_clf_param,
    'ADAboost': ada_clf_param,
    'RandForest': random_forest_clf_param
}


def decision_tree_clf_reg_param(train_data, train_target, test_data, test_target, n=None, k=None):
    clf = DecisionTreeRegressor(min_samples_split=2, max_depth=DT_MAX_REG)
    return calc_accuracy_score(clf=clf, train_data=train_data, train_target=train_target, test_data=test_data,
                               test_target=test_target, is_clf=False)


def random_forest_clf_reg_param(train_data, train_target, test_data, test_target, n=None, k=None):
    clf = RandomForestRegressor(n_estimators=300, max_depth=RF_MAX_DEPTH_REG)
    return calc_accuracy_score(clf=clf, train_data=train_data, train_target=train_target, test_data=test_data,
                               test_target=test_target, is_clf=False)


def knn_clf_reg_param(train_data, train_target, test_data, test_target, n=None, k=None):
    clf = KNeighborsRegressor(n_neighbors=KNN_REG_N)
    return calc_accuracy_score(clf=clf, train_data=train_data, train_target=train_target, test_data=test_data,
                               test_target=test_target, is_clf=False)


def mlp_clf_reg_param(train_data, train_target, test_data, test_target, n=None, k=None):
    clf = MLPRegressor(solver=MLP_SOLVER_REG, max_iter=300, activation=MLP_ACT_REG)
    return calc_accuracy_score(clf=clf, train_data=train_data, train_target=train_target, test_data=test_data,
                               test_target=test_target, is_clf=False)


def ada_clf_reg_param(train_data, train_target, test_data, test_target, n=None, k=None):
    clf = AdaBoostRegressor(n_estimators=ADA_N_EST_REG)
    return calc_accuracy_score(clf=clf, train_data=train_data, train_target=train_target, test_data=test_data,
                               test_target=test_target, is_clf=False)


NAMES_REG_TO_FUNCTIONS_WITH_PARAMS = {
    'DT': decision_tree_clf_reg_param,
    'KNN': knn_clf_reg_param,
    'MLP': mlp_clf_reg_param,
    'ADAboost': ada_clf_reg_param,
    'RandForest': random_forest_clf_reg_param
}
