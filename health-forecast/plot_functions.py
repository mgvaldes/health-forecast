import matplotlib.pyplot as plt
import numpy as np
import itertools


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
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
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
    plt.plot(np.arange(0.1, 1.1, 0.1), cv_scores, color='red', label="CV F1 Score")
    plt.plot(np.arange(0.1, 1.1, 0.1), test_scores, color='blue', label="Test F1 Score")
    plt.xlabel('% of train data')
    plt.ylabel('Scores')
    plt.title('Performance vs. % of data')
    plt.axis('tight')
    plt.legend(loc="upper right")
    plt.show()