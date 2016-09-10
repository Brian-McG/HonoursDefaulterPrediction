from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy import interp
from sklearn.metrics import auc
import sys

almost_black = "#262626"
palette = sns.color_palette()


def visualise_data_set(x_arr, y_arr):
    """Apply principle component analysis to the X-array to 3 dimensions and visualise the resulting matrix"""
    # Instantiate a PCA object for the sake of easy visualisation
    pca = PCA(n_components=3)

    # Fit and transform x to visualise inside a 3D feature space
    x_visualisation = pca.fit_transform(x_arr)

    figure = plt.figure()
    axis = Axes3D(figure)

    axis.scatter(x_visualisation[y_arr == 0, 0], x_visualisation[y_arr == 0, 1], x_visualisation[y_arr == 0, 2],
                 label="Class #0",
                 edgecolor=almost_black, facecolor=palette[0], linewidth=0.3, marker="o")
    axis.scatter(x_visualisation[y_arr == 1, 0], x_visualisation[y_arr == 1, 1], x_visualisation[y_arr == 1, 2],
                 label="Class #1",
                 edgecolor=almost_black, facecolor=palette[2], linewidth=0.3, marker="^")
    axis.set_title("PCA to 3 components")

    plt.show()


def visualise_two_data_sets(x_arr, y_arr, x_arr_two, y_arr_two):
    """Apply principle component analysis to the two X-array"s to 3 dimensions and visualise the resulting matrices"""
    # Instantiate a PCA object for the sake of easy visualisation
    pca = PCA(n_components=3)

    # Fit and transform x to visualise inside a 3D feature space
    x_visualisation = pca.fit_transform(x_arr)

    figure = plt.figure()
    axis = Axes3D(figure)

    axis.scatter(x_visualisation[y_arr == 0, 0], x_visualisation[y_arr == 0, 1], x_visualisation[y_arr == 0, 2],
                 label="Class #0",
                 edgecolor=almost_black, facecolor=palette[0], linewidth=0.3, marker="o")
    axis.scatter(x_visualisation[y_arr == 1, 0], x_visualisation[y_arr == 1, 1], x_visualisation[y_arr == 1, 2],
                 label="Class #1",
                 edgecolor=almost_black, facecolor=palette[2], linewidth=0.3, marker="^")
    axis.set_title("PCA to 3 components - data-set 1")

    x_visualisation_two = pca.transform(x_arr_two)
    figure_two = plt.figure()
    axis_two = Axes3D(figure_two)
    axis_two.scatter(x_visualisation_two[y_arr_two == 0, 0], x_visualisation_two[y_arr_two == 0, 1],
                     x_visualisation_two[y_arr_two == 0, 2],
                     label="Class #0", edgecolor=almost_black,
                     facecolor=palette[0], linewidth=0.3, marker="o")
    axis_two.scatter(x_visualisation_two[y_arr_two == 1, 0], x_visualisation_two[y_arr_two == 1, 1],
                     x_visualisation_two[y_arr_two == 1, 2],
                     label="Class #1", edgecolor=almost_black,
                     facecolor=palette[2], linewidth=0.3, marker="^")
    axis_two.set_title("PCA to 3 components - data-set 2")

    plt.show()


def plot_roc_curve(roc_list, description="classifier"):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure()
    i = 1
    for (tpr, fpr) in roc_list:
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label="ROC fold %d (area = %0.2f)" % (i, roc_auc))
        i += 1

    mean_tpr /= len(roc_list)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, "k--", dashes=[8, 4, 2, 4, 2, 4], label="Mean ROC (area = %0.2f)" % mean_auc, lw=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random selection")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("{0} ROC curve".format(description))
    plt.legend(loc="lower right")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(sys.path[0] + "/../results/{0}_roc_plot_{1}.png".format(description, current_time), bbox_inches="tight")

