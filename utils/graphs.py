import numpy as np
from matplotlib import pyplot as plt
from .utils import SHOW_VALUES

# ------------------------- GRAPH FUNCTIONS -------------------------


def show_values(ax, rects_clf, rects_rgs):
    """
    set values to bar chart
    :param ax: the ax
    :param rects_clf: first bar chart values
    :param rects_rgs: second bar chart values
    :return:
    """
    if SHOW_VALUES:
        for rect in rects_clf:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '%.3f' % float(height),
                    ha='center', va='bottom', color='b')

        for rect in rects_rgs:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '%.2f' % float(height),
                    ha='center', va='bottom', color='r')
    return ax


def show_line_graph_rf(res_clf, res_rgs, first, last, step):
    x = [i for i in range(first, last, step)]
    index = np.arange(len(res_clf))
    plt.title('Random forest - max-depth')

    plt.plot(index, res_rgs, label="reg", color='r', marker='o')
    plt.plot(index, res_clf, label="clf", color='b', marker='o')
    plt.xticks(index, x)
    plt.xlabel('Max depth')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_line_graph_rf_est(res_clf, res_rgs, first, last, step):
    assert len(res_clf) != res_rgs
    x = [i for i in range(first, last, step)]
    plt.title('Random forest - n_estimators')

    plt.plot(x, res_rgs, label="reg", color='r', marker='o')
    plt.plot(x, res_clf, label="clf", color='b', marker='o')
    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_line_graph_dt(res_clf, res_rgs, first, last, step):
    assert len(res_clf) != res_rgs
    index = np.arange(len(res_clf))

    x = [i for i in range(first, last, step)]
    plt.title('DT - best max-depth')

    plt.plot(index, res_rgs, label="reg", color='r', marker='o')
    plt.plot(index, res_clf, label="clf", color='b', marker='o')
    plt.xticks(index, x, fontsize='x-small')
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_line_graph_knn(res_clf, res_rgs, first, last, step):
    assert len(res_clf) != res_rgs
    x = [i for i in range(first, last, step)]
    plt.title('KNN- best n_neighbors')

    plt.plot(x, res_rgs, label="reg", color='r', marker='o')
    plt.plot(x, res_clf, label="clf", color='b', marker='o')
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_bar_graph_mlp(res_clf, res_rgs):
    fig, ax = plt.subplots()
    index = np.arange(len(res_clf))
    width = 0.25
    opacity = 0.8

    rects_clf = plt.bar(index, res_clf, width, alpha=opacity, color='b', label='Classifier')
    rects_rgs = plt.bar(index + width, res_rgs, width, alpha=opacity, color='r', label='Regression')
    show_values(ax=ax, rects_clf=rects_clf, rects_rgs=rects_rgs)

    plt.title('MLP - best solver & activation)')
    plt.xticks(index + width, ('ad+id', 'ad+lg', 'ad+th', 'ad+rl', 'lb+id', 'lb+lg', 'lb+th', 'lb+rl', 'sgd+id',
                               'sgd+lg', 'sgd+th', 'sgd+rl'), fontsize='x-small')
    plt.xlabel('solver + activation')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_line_graph_ada(res_clf, res_rgs, first, last, step):
    x = [i for i in range(first, last, step)]
    plt.title('ADA - best n_estimator')

    plt.plot(x, res_rgs, label="reg", color='r', marker='o')
    plt.plot(x, res_clf, label="clf", color='b', marker='o')
    plt.xlabel('n_estimator')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_line_graph(res_clf, res_rgs, title, dict_list):
    x = dict_list
    plt.title(title)
    plt.tight_layout()
    plt.plot(x, res_rgs, label="reg", color='r', marker='o')
    plt.plot(x, res_clf, label="clf", color='b', marker='o')
    plt.xlabel('K-features')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def show_scatter_graph(res_clf, res_rgs, title):
    x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plt.scatter(x, res_clf, label="clf", color='b')
    plt.scatter(x, res_rgs, label="reg", color='r')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_bar_graph(res_clf, res_rgs, title):
    fig, ax = plt.subplots()
    index = np.arange(len(res_clf))
    bar_width = 0.35
    opacity = 0.8
    rects_clf = plt.bar(index, res_clf, bar_width, alpha=opacity, color='b', label='Classifier')
    rects_rgs = plt.bar(index + bar_width, res_rgs, bar_width, alpha=opacity, color='r', label='Regression')

    show_values(ax=ax, rects_clf=rects_clf, rects_rgs=rects_rgs)

    plt.title(title)
    plt.xticks(index + bar_width, ('DT', 'MLP', 'RandForest', 'KNN', 'ADAboost'))
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_bar_graph_for_DB(res_clf, res_rgs, title):
    fig, ax = plt.subplots()
    index = np.arange(len(res_clf))
    bar_width = 0.25
    opacity = 0.8
    index_width = bar_width/2

    rects_clf = plt.bar(index, res_clf, bar_width, alpha=opacity, color='b', label='Classifier')

    rects_rgs = plt.bar(index + bar_width, res_rgs, bar_width, alpha=opacity, color='r', label='Regression')

    if SHOW_VALUES:
        show_values(ax=ax, rects_clf=rects_clf, rects_rgs=rects_rgs)
    plt.ylim(0, 0.7)
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xticks(index + index_width, ('DEFAULT = TEAM_AVG', 'DEFAULT = 65', 'NO MISSING'))

    plt.legend()
    plt.tight_layout()
    plt.show()


def show_graph_search_graph(list_all: list):
    plt.title("Random-restart hill climbing")
    plt.tight_layout()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    i = 0
    for ll in list_all:
        label = "random " + str(i) if i != 0 else "max"
        x = list(range(len(ll)))
        plt.plot(x, ll, label=label, color=colors[i], marker='o')
        i = i + 1
    plt.xlabel('Steps')
    plt.ylabel('Score')
    plt.legend()
    plt.show()


def show_graph_fitness_bar(list_all: list, list_adv: list):
    fig, ax = plt.subplots()

    plt.title("Random-restart hill climbing")
    plt.tight_layout()
    width = 0.25
    opacity = 0.8
    index = np.arange(len(list_all))
    react = plt.bar(index, list_all, width, alpha=opacity, color='b', label='local max')
    rects_rgs = plt.bar(index + width, list_adv, width, alpha=opacity, color='c', label='local max advanced')
    plt.xticks(index + width, ('max', 'random 1', 'random 2', 'random 3', 'random 4', 'random 5', 'random 6'),
               fontsize='x-small')
    show_values(ax=ax, rects_clf=react, rects_rgs=rects_rgs)
    plt.legend()
    plt.show()


def show_graph_fitness_bar_one_list(list_all: list):
    fig, ax = plt.subplots()
    x = ['max', 'random 1', 'random 2', 'random 3', 'random 4', 'random 5', 'random 6']
    plt.title("Random-restart hill climbing")
    plt.tight_layout()
    width = 0.25
    opacity = 0.8
    react = plt.bar(x, list_all, width, alpha=opacity, color='b', label='local_max')
    show_values(ax=ax, rects_clf=react, rects_rgs=[])
    plt.legend()
    plt.show()


def show_bar_alogrithm_search(list_to_show, title, is_fitness: bool):

    fig, ax = plt.subplots()

    x = ['max xi', 'max_LS', 'random_LS', 'max_LS_Adv', 'random_LS_Adv']
    plt.title(title)
    if is_fitness:
        plt.tight_layout()
    width = 0.25
    opacity = 0.8
    react = plt.bar(x, list_to_show, width, alpha=opacity, color='b')

    show_values(ax=ax, rects_clf=react, rects_rgs=[])
    # plt.legend()
    plt.show()
# ------------------------- END GRAPH FUNCTIONS -------------------------
