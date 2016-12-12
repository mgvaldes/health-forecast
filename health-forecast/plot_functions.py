import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy.interpolate import spline


def plot_confusion_matrix(cm, classes, filename, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, save=True):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')

    fig, ax = plt.subplots()
    ax.xaxis.set_label_position('top')

    if save:
        plt.savefig(filename)
    else:
        plt.show()


def plot_roc(false_rate, true_rate, auc, title, filename, save=True):
    plt.figure()
    lw = 2
    plt.plot(false_rate, true_rate, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    if save:
        plt.savefig(filename)
    else:
        plt.show()


def plot_rf_oob_error(xs, ys, filename, save=True):
    plt.figure()
    plt.plot(xs, ys, color='navy')
    min_index = ys.index(min(ys))
    plt.axvline(x=xs[min_index], color='red')
    plt.xlabel("NUM_TREES")
    plt.ylabel("OOB error rate")
    if save:
        plt.savefig(filename)
    else:
        plt.show()


def plot_metrics_vs_data(cv_scores, test_scores):
    plt.figure()
    plt.plot(np.arange(0.3, 1.1, 0.1), cv_scores, color='red', label="CV F1 Score")
    plt.plot(np.arange(0.3, 1.1, 0.1), test_scores, color='blue', label="Test F1 Score")
    plt.xlabel('% of train data')
    plt.ylabel('Scores')
    plt.title('Performance vs. % of data')
    plt.axis('tight')
    plt.legend(loc="upper right")
    plt.show()


def plot_prob_vs_frequency(y_prob, y_test):
    # def plot_prob_vs_frequency(y_prob):
    # freq_0 = []
    # freq_1 = []
    #
    # for i in range(1, len(y_test) + 1):
    #     freq_0.append((y_test[:i] == 0).sum())
    #     freq_1.append((y_test[:i] == 1).sum())
    #
    # plt.figure()
    #
    # print(y_prob[:, 1])
    # print(np.sort(y_prob[:, 1][::-1]))
    #
    # plt.plot(np.sort(y_prob[:, 1][::-1]), freq_0, color='red', label="TP")
    # plt.plot(np.sort(y_prob[:, 1][::-1]), freq_1, color='blue', label="TN")
    # plt.xlabel('Test value')
    # plt.ylabel('Frequency')
    # plt.axis('tight')
    # plt.legend(loc="upper right")
    # plt.show()

    y_test_and_y_prob = np.zeros((len(y_test), 2))
    y_test_and_y_prob[:, 0] = y_test
    y_test_and_y_prob[:, 1] = y_prob[:, 0]
    # print(y_test_and_y_prob)

    sorted_by_y_prob_y_test_and_y_prob = y_test_and_y_prob[np.argsort(y_test_and_y_prob[:, 1])[::-1]]
    print(sorted_by_y_prob_y_test_and_y_prob)

    acum_tps = np.zeros(len(y_test))

    for i in range(0, len(y_test)):
        acum_tps[i] = sum(sorted_by_y_prob_y_test_and_y_prob[:, 0][np.where(sorted_by_y_prob_y_test_and_y_prob[:, 1][0:(i + 1)] >= 0.5)])

    print(acum_tps)

    plt.figure()
    plt.plot(range(1, len(y_test) + 1), acum_tps, color='red')
    plt.xlabel('Num. Samples')
    plt.ylabel('TP')
    plt.axis('tight')
    plt.show()

    # plt.figure()
    #
    # xnew = np.linspace(0, len(y_test), 30)
    #
    # TP_smooth = spline(range(0, len(y_test)), acum_tps, xnew)
    #
    # plt.plot(xnew, TP_smooth)
    # plt.xlabel('Num. Samples')
    # plt.ylabel('TP')
    # plt.axis('tight')
    # plt.show()

    # sorted_y_prob = np.sort(y_prob[:, 0])[::-1]
    # print(sorted_y_prob)
    # min_prob = min(sorted_y_prob)
    # max_prob = max(sorted_y_prob)
    # binwidth = 0.1

    # plt.figure()
    #
    # bins = np.arange(0, 1.1, 0.1)
    # plt.hist(sorted_y_prob, bins)

    # plt.xticks(np.arange(0, 1, 0.1))
    # plt.xlim([-0.05, 1.05])

    # his = plt.hist(sorted_y_prob)

    # ax = plt.gca()
    # ax.invert_xaxis()

    # his = np.histogram(y_prob)
    # fig, ax = plt.subplots()
    # offset = .5
    # plt.bar(his[1][1:], his[0], width=1)
    # ax.set_xticks(his[1][1:] + offset)
    # ax.set_xticklabels(('1', '2', '3', '4'))

    # plt.show()