import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cycler import cycler
from lifelines.estimation import KaplanMeierFitter
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import interp
from sklearn.decomposition import PCA
from sklearn.metrics import auc
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import math

from config import constants as const

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


def plot_roc_curve_of_classifier(roc_list, data_set_description, classifier_description="classifier"):
    if const.RECORD_RESULTS is True and not (None, None) in roc_list:
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        fig = plt.figure(figsize=(12, 10))
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
        plt.plot([0, 1], [0, 1], "k--", label="Random classification")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("{0} ROC curve".format(classifier_description))
        plt.legend(loc="lower right")
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/{0}_{1}_roc_plot_{2}.png".format(data_set_description, classifier_description, current_time),
                    bbox_inches="tight")
        plt.close(fig)


def plot_mean_roc_curve_of_balancers(balancer_roc_list, data_set_description, classifier_description):
    if const.RECORD_RESULTS is True and not (None, None) in balancer_roc_list[0][0]:
        fig = plt.figure(figsize=(12, 10))
        monochrome = (cycler("color", ["k"]) * cycler("marker", [""]) *
                      cycler("linestyle", ["-", "--", "-."]))
        color = iter(cm.brg(np.linspace(0, 1, len(balancer_roc_list))))
        plt.rc("axes", prop_cycle=monochrome)

        for (test_run_roc_list, balancer) in balancer_roc_list:
            mean_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 100)
            for test_result in test_run_roc_list:
                for (tpr, fpr) in test_result:
                    mean_tpr += interp(mean_fpr, fpr, tpr)
                    mean_tpr[0] = 0.0

            mean_tpr /= (len(test_result) * len(test_run_roc_list))
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            c = next(color)
            plt.plot(mean_fpr, mean_tpr, c=c, lw=1, alpha=0.7, label="{0} (area = {1:.4f})".format(balancer, mean_auc))

        plt.plot([0, 1], [0, 1], "k--", label="Random classification")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("{0} ROC curve for each balancer".format(classifier_description))
        plt.legend(loc="lower right")
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(
            os.path.dirname(os.path.realpath(__file__)) + "/../results/{0}_{1}_roc_balancer_plot_{2}.png".format(data_set_description, classifier_description, current_time),
            bbox_inches="tight")
        plt.close(fig)


def plot_mean_roc_curve_of_classifiers(classifier_roc_list, data_set_description):
    if const.RECORD_RESULTS is True:
        fig = plt.figure(figsize=(8, 6.66))
        monochrome = (cycler("color", ["k"]) * cycler("marker", [""]) *
                      cycler("linestyle", ["-", "--", "-."]))
        color_arr = ["#64B3DE", "#1f78b4", "#6ABF20", "#FBAC44", "#bc1659", "#B9B914", "#33a02c", "#ff7f00", "#6a3d9a", "black", "#b15928", "#e31a1c"]
        plt.rc("axes", prop_cycle=monochrome)
        line_style_index = 0
        color_index = 0

        for (test_run_roc_list, classifier_description) in classifier_roc_list:
            if not (None, None) in test_run_roc_list[0]:
                mean_tpr = 0.0
                mean_fpr = np.linspace(0, 1, 100)
                count = 0
                for roc_list in test_run_roc_list:
                    for (tpr, fpr) in roc_list:
                        mean_tpr += interp(mean_fpr, fpr, tpr)
                        mean_tpr[0] = 0.0
                        count += 1

                mean_tpr /= float(count)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)
                line_width = 0.5
                if line_style_index == 1:
                    line_width = 0.8
                elif line_style_index == 2:
                    line_width = 1.5

                plt.plot(mean_fpr, mean_tpr, c=color_arr[color_index], lw=line_width, alpha=1, label="{0} ({1:.3f})".format(classifier_description, mean_auc))
                line_style_index = (line_style_index + 1) % 3
                color_index += 1

        plt.locator_params(axis='x', nbins=10)
        plt.locator_params(axis='y', nbins=10)
        plt.plot([0, 1], [0, 1], "k--", label="Random classification", lw=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve for each classifier")
        plt.legend(loc="lower right", fancybox=True, frameon=True)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/{0}_roc_classifier_plot_{1}.png".format(data_set_description, current_time), bbox_inches="tight")
        plt.close(fig)


def plot_balancer_results_per_classifier(data_balancer_results_per_classifier, parameter=(2,"Balanced Accuracy")):
    classifier_arr = []
    color = iter(cm.Set1(np.linspace(0, 1, len(data_balancer_results_per_classifier) + 1)))
    mean_classifier_arr = [0] * len(data_balancer_results_per_classifier[0][1])
    for (classifier_name, data_balancer_results) in data_balancer_results_per_classifier:
        individual_data_balance_plot = []
        x = 0
        for (data_balancer_name, result_arr) in data_balancer_results:
            individual_data_balance_plot.append(result_arr[parameter[0]])  # Average True rate
            mean_classifier_arr[x] += result_arr[parameter[0]]
            x += 1
        classifier_arr.append(individual_data_balance_plot)

    classifier_arr.append([value / float(len(data_balancer_results_per_classifier)) for value in mean_classifier_arr])

    fig = plt.figure(figsize=(12, 10))

    classifiers = np.arange(len(classifier_arr))
    data_balancers = np.arange(len(classifier_arr[0])) * 3
    bar_width = 0.2
    opacity = 0.9

    for i in range(len(classifier_arr)):
        if i + 1 != len(classifier_arr):
            label = data_balancer_results_per_classifier[i][0]
        else:
            label = "Mean classification"

        plt.bar(data_balancers + (i * bar_width), classifier_arr[i], bar_width,
                alpha=opacity,
                color=color.next(),
                label=label)

    plt.locator_params(axis='y', nbins=10)
    plt.xlabel("Data balance algorithm")
    plt.ylabel(parameter[1])
    plt.legend(loc="lower right", fancybox=True, frameon=True)
    plt.title("{0} per data balance algorithm".format(parameter[1]))
    plt.ylim([0.0, 1.00])
    data_balance_labels = [filter(str.isupper, data_balance_name) if data_balance_name != "None" and len(filter(str.isupper, data_balance_name)) < 6 else data_balance_name for
                           (data_balance_name, _) in data_balancer_results_per_classifier[0][1]]
    plt.xticks(data_balancers + (bar_width / 2) * len(classifiers), data_balance_labels)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/data_balancer_results_per_classifier_plot_{0}_{1}.png".format(parameter[1], current_time))
    plt.close(fig)


def plot_kaplan_meier_graph_of_time_to_default(time_to_default, data_set_description=""):
    kmf = KaplanMeierFitter()
    kmf.fit(time_to_default, event_observed=[1] * len(time_to_default))
    ax = kmf.plot(title="{0} Kaplan Meier analysis of time to default".format(data_set_description).replace("_", " "))
    ax.get_figure().set_size_inches(12, 10)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.ylim([0.0, 1.00])
    plt.ylabel("Percentage Remaining")
    plt.xlabel("Time to default (days)")
    plt.locator_params(axis='x', nbins=10)

    ax.get_figure().savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/kaplan_meier_time_to_default_{0}.png".format(current_time), bbox_inches="tight")
    plt.close(ax.get_figure())


def plot_percentage_difference_graph(results, datasets, name_suffix="", parameter="Balanced Accuracy", x_label="Feature selection approach", difference_from="no feature selection"):
    patterns = (None, "////")

    colors = ["#64B3DE", "#1f78b4", "#B9B914", "#FBAC44", "#bc1659", "#33a02c", "#6ABF20", "#ff7f00", "#6a3d9a", "#5a2add", "#b15928", "#e31a1c", "grey"]
    classifier_arr = []
    for i in range(len(results)):
        classifier_arr.append(list())
    index = 0
    for results_per_classifier in results:
        no_feature_selection = results[index][0][1]
        for i in range(len(no_feature_selection) + 1):
            classifier_arr[index].append(list())
        for i in range(1, len(results_per_classifier)):
            data_balancer_results = results_per_classifier[i][1]
            x = 0
            mean_classification = 0
            for (result_arr, data_balancer_name, _) in data_balancer_results:
                value = result_arr[2] - no_feature_selection[x][0][2]
                classifier_arr[index][x].append(value)
                mean_classification += value
                x += 1
            mean_classification /= float(len(data_balancer_results))
            classifier_arr[index][x].append(mean_classification)
        index += 1

    fig = plt.figure(figsize=(16, 5))

    classifiers = np.arange(len(classifier_arr[0]))
    data_balancers = np.arange(len(classifier_arr[0][0])) * 3
    bar_width = 0.2
    opacity = 0.9
    subplt_val = 101 + (10 * len(results))
    ax1 = plt.subplot(subplt_val)
    for i in range(len(classifier_arr[0])):
        if i + 1 != len(classifier_arr[0]):
            label = results[0][0][1][i][1]
        else:
            label = "Mean classification"
        plt.bar(data_balancers + (i * bar_width), classifier_arr[0][i], bar_width,
                alpha=opacity,
                color=colors[i],
                hatch=patterns[i % len(patterns)],
                label=label)

    legend = plt.legend(loc='lower center', bbox_to_anchor=(len(results) / 2.0, -0.31), fancybox=True, frameon=True, ncol=6)
    legend.get_frame().set_facecolor('#ffffff')

    plt.xlabel(x_label, x=len(results) / 2.0)
    plt.ylabel("Difference in {0} from {1}".format(parameter, difference_from))
    feature_selection_labels = [results[0][i][0] for i in range(1, len(results[0]))]
    plt.xticks(data_balancers + (bar_width / 2) * len(classifiers), feature_selection_labels)
    plt.title(datasets[0].replace("_", " "))

    for z in range(1, len(results)):
        ax2 = plt.subplot(subplt_val + z, sharey=ax1)
        color = iter(cm.Set1(np.linspace(0, 1, len(no_feature_selection) + 1)))
        for i in range(len(classifier_arr[z])):
            if i + 1 != len(classifier_arr[z]):
                label = results[z][0][1][i][1]
            else:
                label = "Mean classification"
            plt.bar(data_balancers + (i * bar_width), classifier_arr[z][i], bar_width,
                    alpha=opacity,
                    color=colors[i],
                    hatch=patterns[i % len(patterns)],
                    label=label)

        feature_selection_labels = [results[z][i][0] for i in range(1, len(results[z]))]
        plt.xticks(data_balancers + (bar_width / 2) * len(classifiers), feature_selection_labels)
        plt.title(datasets[z].replace("_", " "))

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    plt.locator_params(axis='y', nbins=15)
    name = "{3}_results_per_classifier_plot{0}_{4}_{1}_{2}".format(name_suffix, parameter, current_time, x_label, datasets)
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/{0}".format(name.replace(" ", "_")), bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)


def plot_time_to_default_results(time_to_default_results_per_classifier, parameter="Difference in balanced accuracy from no feature selection"):
    color = iter(cm.Set1(np.linspace(0, 1, len(time_to_default_results_per_classifier[0][1]) + 1)))
    classifier_arr = []
    for i in range(len(time_to_default_results_per_classifier[0][1]) + 1):
        classifier_arr.append(list())

    for i in range(0, len(time_to_default_results_per_classifier)):
        data_balancer_results = time_to_default_results_per_classifier[i][1]
        x = 0
        mean_classification = 0
        for (result_arr, data_balancer_name, _) in data_balancer_results:
            result = result_arr[2]
            classifier_arr[x].append(result)
            mean_classification += result
            x += 1
        mean_classification /= float(len(data_balancer_results))
        classifier_arr[x].append(mean_classification)

    fig = plt.figure(figsize=(12, 10))

    classifiers = np.arange(len(classifier_arr))
    data_balancers = np.arange(len(classifier_arr[0])) * 3
    bar_width = 0.2
    opacity = 0.9

    for i in range(len(classifier_arr)):
        if i == len(classifier_arr) - 1:
            label = "Mean classification"
        else:
            label = time_to_default_results_per_classifier[0][1][i][1]
        plt.bar(data_balancers + (i * bar_width), classifier_arr[i], bar_width,
                alpha=opacity,
                color=color.next(),
                label=label)

    plt.locator_params(axis='y', nbins=10)
    plt.xlabel("Default range (days)")
    plt.ylabel(parameter)
    plt.ylim([0.0, 1.00])
    plt.legend(loc="lower right", fancybox=True, frameon=True)
    plt.title("{0} when trained on different default ranges".format(parameter))
    feature_selection_labels = [time_to_default_results_per_classifier[i][0] for i in range(0, len(time_to_default_results_per_classifier))]
    plt.xticks(data_balancers + (bar_width / 2) * len(classifiers), feature_selection_labels)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/time_to_default_results_per_classifier_plot_{0}_{1}.png".format(parameter, current_time))
    plt.close(fig)


def visualise_dataset_classifier_results(dataset_results):
    sns.set(style='ticks')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    markers = ["s", "o", "^", "*"]
    colors = ["#64B3DE", "#1f78b4", "#6ABF20", "#FBAC44", "#bc1659", "#B9B914", "#33a02c", "#ff7f00", "#6a3d9a", "black", "#b15928", "#e31a1c"]

    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_axis_on()
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    plt.xlabel("Difference in TPR")
    plt.ylabel("Difference in TNR")
    plt.title("Comparision of TNR toTPR across data sets for each classifier", y=1.03, fontsize=16)

    ax.xaxis.set_label_coords(0.1, 0.52)
    ax.yaxis.set_label_coords(0.53, 0.9)

    plt.ylim(-0.2, 0.2)
    plt.xlim(-0.2, 0.2)
    data_set_labels = []
    classifier_labels = []
    data_set_index = 0
    for (data_set, dataset_result) in dataset_results:
        data_set_labels.append(mlines.Line2D(range(1), range(1), color="white", marker=markers[data_set_index], markeredgecolor="black", markeredgewidth=1.0, label=data_set.replace("_", " ")))
        median_true_pos = np.median(np.array([result_arr[3] for (result_arr, classifier_description) in dataset_result]))
        median_true_neg = np.median(np.array([result_arr[4] for (result_arr, classifier_description) in dataset_result]))

        i = 0
        for (result_arr, classifier_description) in dataset_result:
            if data_set_index == 0:
                classifier_labels.append(mpatches.Patch(color=colors[i], label=classifier_description, alpha=0.8))
            ax.scatter(result_arr[3] - median_true_pos, result_arr[4] - median_true_neg, marker=markers[data_set_index], s=100, alpha=0.8, color=colors[i], edgecolor="white", zorder=data_set_index, lw=0)
            i += 1
        data_set_index += 1

    plt.legend(handles=data_set_labels + classifier_labels)
    sns.despine()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/classifier_dataset_plt_{0}.png".format(current_time), bbox_inches='tight')
    plt.close(fig)


def visualise_dataset_balancer_results(dataset_results):
    sns.set(style='ticks')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    markers = ["s", "o", "^", "*", "v", "^", "<", ">", "H", "d"]
    colors = ["#64B3DE", "#1f78b4", "#6ABF20", "#FBAC44", "#bc1659", "#B9B914", "#33a02c", "#ff7f00", "#6a3d9a", "black", "#b15928", "#e31a1c"]
    hatches = ["///////", "*", "."]

    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_axis_on()
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    plt.xlabel("Difference in TPR")
    plt.ylabel("Difference in TNR")
    plt.title("Comparision of TNR toTPR across data sets for each classifier", y=1.03, fontsize=16)

    ax.xaxis.set_label_coords(0.1, 0.52)
    ax.yaxis.set_label_coords(0.53, 0.9)

    plt.ylim(-0.2, 0.2)
    plt.xlim(-0.2, 0.2)
    data_set_labels = []
    classifier_labels = []
    data_set_index = 0
    for (data_set, dataset_result) in dataset_results:
        data_set_labels.append(mlines.Line2D(range(1), range(1), color="white", marker=markers[data_set_index], markeredgecolor="black", markeredgewidth=1.0, label=data_set.replace("_", " ")))
        true_pos_arr = []
        true_neg_arr = []

        for (classifier_description, result_arr) in dataset_result:
            for (balancer_description, results) in result_arr:
                true_pos_arr.append(results[3])
                true_neg_arr.append(results[4])

        median_true_pos = np.median(np.array(true_pos_arr))
        median_true_neg = np.median(np.array(true_neg_arr))

        i = 0
        for (classifier_description, result_arr) in dataset_result:
            balancer_index = 0
            for (balancer_description, results) in result_arr:
                if data_set_index == 0:
                    classifier_labels.append(mpatches.Patch(color=colors[i], label=classifier_description, alpha=0.8))
                ax.scatter(results[3] - median_true_pos, results[4] - median_true_neg, marker=markers[data_set_index], s=100, alpha=0.8, color=colors[i], edgecolor="white", zorder=data_set_index, lw=0)
                balancer_index += 1
            i += 1
        data_set_index += 1

    plt.legend(handles=data_set_labels + classifier_labels)
    sns.despine()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/classifier_dataset_plt_{0}.png".format(current_time), bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    results = [[('Default without balancer', [([0.28510978429582373, 0.2513911144588335, 0.592276699366446, 0.20809116809116807, 0.9764622306417239, 0.023537769358276103, 0.7919088319088318, 0.0, 0.00019417475728155338, 0.0, 0.0, 0.0, 1.1018989431868618], 'AdaBoost', 'Default without balancer'), ([0.24182839366076131, 0.22163086580603952, 0.5848291758670636, 0.20301994301994303, 0.9666384087141842, 0.03336159128581577, 0.7969800569800569, 0.09692307692307692, 0.9257759886336727, 0.016241060857210513, 0.6455555555555557, 0.6455555555555557, 4.830525448338107], 'Artificial neural network', 'Default without balancer'), ([0.36034278499101369, 0.34905040372439144, 0.7156847382364786, 0.5595726495726495, 0.8717968269003078, 0.12820317309969215, 0.4404273504273505, 0.5252421652421652, 0.8488401610229694, 0.11225479516931092, 0.41803418803418796, 0.41803418803418796, 0.02413255896087598], 'Bernoulli Naive Bayes', 'Default without balancer'), ([0.21441356916752091, 0.21296288659151327, 0.5983998467218927, 0.26937321937321934, 0.9274264740705661, 0.07257352592943404, 0.7306267806267807, 0.2544444444444444, 0.8909476675349278, 0.06994506275159838, 0.6411680911680911, 0.6411680911680911, 0.021761657092661972], 'Decision Tree', 'Default without balancer'), ([0.13191996526052396, 0.077290814365959221, 0.5241398338902483, 0.055185185185185184, 0.9930944825953114, 0.006905517404688611, 0.944814814814815, 0.0037037037037037034, 0.9312237745678429, 0.00019417475728155338, 0.6686324786324785, 0.6686324786324785, 0.17888664841417667], 'Extreme Learning Machine', 'Default without balancer'), ([0.06339420861943193, 0.013714102044819775, 0.5275653007239557, 0.9649572649572649, 0.09017333649064646, 0.9098266635093536, 0.03504273504273504, 0.9634757834757833, 0.08725361117688848, 0.906323940326782, 0.03356125356125356, 0.03356125356125356, 0.006997938293704209], 'Gaussian Naive Bayes', 'Default without balancer'), ([0.21663861085834971, 0.1639037419140307, 0.5545965142723371, 0.12310541310541309, 0.9860876154392614, 0.013912384560738811, 0.876894586894587, 0.026096866096866095, 0.9390073407530192, 0.0017518351882547952, 0.6768376068376067, 0.6768376068376067, 0.012492128990971274], 'K-nearest neighbours', 'Default without balancer'), ([0.2938485180612182, 0.25196153949978162, 0.5902653026804103, 0.19988603988603987, 0.980644565474781, 0.01935543452521904, 0.8001139601139601, 0.050626780626780624, 0.9382334833057067, 0.00359838977030547, 0.6015384615384616, 0.6015384615384616, 0.022914431642211023], 'Logistic regression', 'Default without balancer'), ([0.20432077958226419, 0.19265372591987179, 0.577003499355382, 0.19777777777777775, 0.9562292209329859, 0.043770779067013973, 0.8022222222222222, 0.06786324786324786, 0.8842367984844897, 0.015560975609756094, 0.585042735042735, 0.585042735042735, 0.30549956381177223], 'Random forest', 'Default without balancer'), ([0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1545811234639079], 'SVM (RDF)', 'Default without balancer'), ([0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15337467814138545], 'SVM (linear)', 'Default without balancer'), ([0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.18373202044198397], 'SVM (polynomial)', 'Default without balancer')], 'Default without balancer'), ('Default with balancer', [([0.30153118118721634, 0.26383586308512585, 0.7100914581861776, 0.6445868945868946, 0.7755960217854605, 0.22440397821453942, 0.3554131054131054, 0.0, 0.00019417475728155338, 0.0, 0.0, 0.0, 1.2448997974516944], 'AdaBoost', 'Default with balancer'), ([0.20842361343637891, 0.20169332433254286, 0.6250144251430066, 0.40071225071225064, 0.8493165995737628, 0.15068340042623724, 0.5992877492877493, 0.2791737891737891, 0.7423173099692161, 0.06828842055410847, 0.4537037037037036, 0.4537037037037036, 6.4462046794393855], 'Artificial neural network', 'Default with balancer'), ([0.3568124955743508, 0.34465777204389991, 0.7151070396613848, 0.5618233618233618, 0.8683907174994079, 0.13160928250059198, 0.4381766381766382, 0.5304558404558405, 0.8469902912621359, 0.11468624200805115, 0.4150427350427351, 0.4150427350427351, 0.017092088541683027], 'Bernoulli Naive Bayes', 'Default with balancer'), ([0.19123044132835967, 0.14848196873461436, 0.6439562840313491, 0.6207977207977209, 0.6671148472649776, 0.33288515273502256, 0.3792022792022792, 0.6162393162393163, 0.5784224484963297, 0.32783139947904333, 0.33361823361823356, 0.33361823361823356, 0.0066653183570343725], 'Decision Tree', 'Default with balancer'), ([0.2366144484941354, 0.19645827879101313, 0.6716276961126594, 0.6206837606837607, 0.7225716315415582, 0.2774283684584419, 0.3793162393162393, 0.366980056980057, 0.36449775041439725, 0.09512289841345015, 0.141994301994302, 0.141994301994302, 0.052529770978540835], 'Extreme Learning Machine', 'Default with balancer'), ([0.17210035096987791, 0.10846262097567863, 0.6319205470247382, 0.76008547008547, 0.5037556239640067, 0.49624437603599325, 0.2399145299145299, 0.757094017094017, 0.49071228984134496, 0.48553350698555525, 0.23005698005698005, 0.23005698005698005, 0.012249403071127762], 'Gaussian Naive Bayes', 'Default with balancer'), ([0.25846779041275014, 0.23668874067400206, 0.6711969144685223, 0.5364672364672365, 0.8059265924698081, 0.1940734075301918, 0.4635327635327635, 0.39398860398860397, 0.7161539190149182, 0.12413023916646934, 0.36190883190883183, 0.36190883190883183, 0.023738419881731688], 'K-nearest neighbours', 'Default with balancer'), ([0.30094294823347278, 0.26346299136136053, 0.709518422045062, 0.642962962962963, 0.7760738811271607, 0.22392611887283925, 0.35703703703703693, 0.45592592592592596, 0.4995136159128581, 0.09065593180203646, 0.15150997150997153, 0.15150997150997153, 0.051727275197039194], 'Logistic regression', 'Default with balancer'), ([0.22793459193961066, 0.17937137152052474, 0.6705498238178796, 0.655982905982906, 0.6851167416528534, 0.31488325834714664, 0.344017094017094, 0.45131054131054127, 0.42584844896992663, 0.1596206488278475, 0.17663817663817666, 0.17663817663817666, 0.25866689406814086], 'Random forest', 'Default with balancer'), ([0.32489823690777381, 0.30794950723974723, 0.703690163013156, 0.5617378917378917, 0.8456424342884205, 0.15435756571157946, 0.43826210826210826, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01786699428578321], 'SVM (RDF)', 'Default with balancer'), ([0.33630414463437025, 0.32167548664777268, 0.7072785741897748, 0.5588034188034187, 0.8557537295761307, 0.1442462704238693, 0.4411965811965811, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7888687480945069], 'SVM (linear)', 'Default with balancer'), ([0.30590075588642013, 0.28204436396053306, 0.7006562900356411, 0.5828205128205128, 0.8184920672507696, 0.1815079327492304, 0.4171794871794871, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8532456328351632], 'SVM (polynomial)', 'Default with balancer')], 'Default with balancer'), ('Tuned', [([0.2974159083987617, 0.25587234056440289, 0.7103271037116644, 0.658974358974359, 0.76167984844897, 0.2383201515510301, 0.34102564102564104, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6331921733644453], 'AdaBoost', 'Tuned'), ([0.30356828882979892, 0.26685461806121913, 0.7104998424716634, 0.6414245014245015, 0.7795751835188255, 0.22042481648117454, 0.3585754985754986, 0.46336182336182336, 0.49872981292919727, 0.091723892967085, 0.1559259259259259, 0.1559259259259259, 3.3973560352452155], 'Artificial neural network', 'Tuned'), ([0.30993405853653133, 0.28155491948262473, 0.7074697272364807, 0.6081196581196581, 0.8068197963533033, 0.1931802036466967, 0.3918803418803419, 0.5894871794871794, 0.7557404688609993, 0.16147099218565003, 0.3493732193732194, 0.3493732193732194, 0.011355262482680928], 'Bernoulli Naive Bayes', 'Tuned'), ([0.24006576460348664, 0.21108380101454244, 0.6661235055890514, 0.5641025641025641, 0.7681444470755386, 0.2318555529244613, 0.43589743589743585, 0.4276923076923077, 0.43098839687426, 0.11584608098508171, 0.18914529914529915, 0.18914529914529915, 0.003967290410801597], 'Decision Tree', 'Tuned'), ([0.2366144484941354, 0.19645827879101313, 0.6716276961126594, 0.6206837606837607, 0.7225716315415582, 0.2774283684584419, 0.3793162393162393, 0.366980056980057, 0.36449775041439725, 0.09512289841345015, 0.141994301994302, 0.141994301994302, 0.036617775349479006], 'Extreme Learning Machine', 'Tuned'), ([0.17210035096987791, 0.10846262097567863, 0.6319205470247382, 0.76008547008547, 0.5037556239640067, 0.49624437603599325, 0.2399145299145299, 0.757094017094017, 0.49071228984134496, 0.48553350698555525, 0.23005698005698005, 0.23005698005698005, 0.010134961769384176], 'Gaussian Naive Bayes', 'Tuned'), ([0.23164052619626113, 0.18330227536259722, 0.6730145317360567, 0.6575213675213675, 0.6885076959507461, 0.31149230404925404, 0.3424786324786325, 0.3572934472934473, 0.12432157234193701, 0.07976651669429316, 0.033504273504273506, 0.033504273504273506, 0.0021888196957879913], 'K-nearest neighbours', 'Tuned'), ([0.30099449913856979, 0.26269055944633141, 0.7101695058872421, 0.6466951566951568, 0.7736438550793274, 0.2263561449206725, 0.35330484330484324, 0.4581481481481481, 0.49814870944825945, 0.0917262609519299, 0.15074074074074073, 0.15074074074074073, 0.06163517759954602], 'Logistic regression', 'Tuned'), ([0.27714859091340671, 0.24047725557322591, 0.6943100602925372, 0.6246153846153847, 0.7640047359696898, 0.23599526403031018, 0.37538461538461537, 0.4201424501424501, 0.34658868103244134, 0.08618138763911912, 0.1094871794871795, 0.1094871794871795, 0.34802305609471207], 'Random forest', 'Tuned'), ([0.28541237832898281, 0.24767054889395204, 0.7002980861150409, 0.6343304843304843, 0.7662656878995975, 0.2337343121004026, 0.36566951566951567, 0.4259829059829059, 0.36438313994790433, 0.08773052332465073, 0.11626780626780626, 0.11626780626780626, 0.09316385886216734], 'SVM (RDF)', 'Tuned'), ([0.29921962167234822, 0.26852602610407844, 0.7030687862492265, 0.610997150997151, 0.7951404215013023, 0.2048595784986976, 0.389002849002849, 0.4455840455840455, 0.4864797537295762, 0.08676627989580868, 0.13586894586894588, 0.13586894586894588, 7.3027357906779455], 'SVM (linear)', 'Tuned'), ([0.32827699530620075, 0.31297721264090528, 0.7036719888981314, 0.5557834757834758, 0.8515605020127872, 0.14843949798721287, 0.4442165242165242, 0.47148148148148145, 0.7218820743547242, 0.09503954534690975, 0.3125071225071224, 0.3125071225071224, 2.721206686810815], 'SVM (polynomial)', 'Tuned')], 'Tuned')], [('Default without balancer', [([0.35857594690417915, 0.35505373641054433, 0.6682619047619047, 0.48066666666666674, 0.8558571428571428, 0.14414285714285716, 0.5193333333333333, 0.0, 0.0007142857142857143, 0.0, 0.0, 0.0, 1.3222362721452314], 'AdaBoost', 'Default without balancer'), ([0.37887981926509434, 0.3772550041717963, 0.6819761904761904, 0.5176666666666666, 0.8462857142857143, 0.1537142857142857, 0.4823333333333334, 0.4093333333333334, 0.7705714285714286, 0.09899999999999999, 0.373, 0.373, 5.956481545766559], 'Artificial neural network', 'Default without balancer'), ([0.38143090742111829, 0.38123031961713377, 0.693047619047619, 0.5836666666666667, 0.8024285714285714, 0.19757142857142856, 0.4163333333333334, 0.41600000000000004, 0.703857142857143, 0.11885714285714286, 0.2803333333333333, 0.2803333333333333, 0.014382620087599562], 'Bernoulli Naive Bayes', 'Default without balancer'), ([0.261269595095667, 0.26112175447055586, 0.6318333333333334, 0.4946666666666667, 0.7689999999999999, 0.23099999999999996, 0.5053333333333334, 0.4946666666666667, 0.7689999999999999, 0.23099999999999996, 0.5053333333333334, 0.5053333333333334, 0.04768310095116203], 'Decision Tree', 'Default without balancer'), ([0.22729057607184183, 0.20126420135539419, 0.585452380952381, 0.2533333333333333, 0.9175714285714285, 0.08242857142857142, 0.7466666666666667, 0.024666666666666663, 0.5977142857142856, 0.006428571428571428, 0.30266666666666664, 0.30266666666666664, 0.15508218148448902], 'Extreme Learning Machine', 'Default without balancer'), ([0.31863766794126008, 0.29687183955021978, 0.6736190476190476, 0.7216666666666667, 0.6255714285714287, 0.37442857142857144, 0.2783333333333333, 0.6823333333333332, 0.5858571428571429, 0.33571428571428574, 0.231, 0.231, 0.008863646654331703], 'Gaussian Naive Bayes', 'Default without balancer'), ([0.30384456393479498, 0.28423084970733375, 0.6245238095238095, 0.3443333333333334, 0.9047142857142857, 0.09528571428571428, 0.6556666666666666, 0.11733333333333333, 0.7315714285714285, 0.023, 0.3866666666666666, 0.3866666666666666, 0.011756948070961393], 'K-nearest neighbours', 'Default without balancer'), ([0.38228194598877463, 0.37731814201323194, 0.6774047619047618, 0.48566666666666664, 0.869142857142857, 0.13085714285714287, 0.5143333333333333, 0.21533333333333332, 0.724, 0.03685714285714285, 0.29233333333333333, 0.29233333333333333, 0.044268124450991334], 'Logistic regression', 'Default without balancer'), ([0.27978112841763964, 0.25834168824177012, 0.6120476190476192, 0.31766666666666665, 0.9064285714285715, 0.09357142857142856, 0.6823333333333332, 0.09033333333333333, 0.5708571428571428, 0.015428571428571427, 0.2273333333333333, 0.2273333333333333, 0.28667805686445436], 'Random forest', 'Default without balancer'), ([0.2939581548817165, 0.24581694117869435, 0.6008333333333333, 0.2526666666666667, 0.9490000000000001, 0.051000000000000004, 0.7473333333333334, 0.0, 0.0, 0.0, 0.0, 0.0, 0.228961026808219], 'SVM (RDF)', 'Default without balancer'), ([0.2939581548817165, 0.24581694117869435, 0.6008333333333333, 0.2526666666666667, 0.9490000000000001, 0.051000000000000004, 0.7473333333333334, 0.0, 0.0, 0.0, 0.0, 0.0, 0.27745404930587225], 'SVM (linear)', 'Default without balancer'), ([0.2939581548817165, 0.24581694117869435, 0.6008333333333333, 0.2526666666666667, 0.9490000000000001, 0.051000000000000004, 0.7473333333333334, 0.0, 0.0, 0.0, 0.0, 0.0, 0.22753057090850898], 'SVM (polynomial)', 'Default without balancer')], 'Default without balancer'), ('Default with balancer', [([0.38488722479138365, 0.37402041597088626, 0.7064285714285714, 0.698, 0.7148571428571429, 0.28514285714285714, 0.302, 0.0006666666666666666, 0.0007142857142857143, 0.0004285714285714286, 0.0, 0.0, 1.2253275335895102], 'AdaBoost', 'Default with balancer'), ([0.38227343399761854, 0.36954706425987122, 0.7057142857142857, 0.7070000000000001, 0.7044285714285714, 0.2955714285714286, 0.293, 0.5953333333333333, 0.6064285714285715, 0.21314285714285713, 0.21033333333333334, 0.21033333333333334, 5.285892586007562], 'Artificial neural network', 'Default with balancer'), ([0.39331384981642808, 0.37738257655042801, 0.7126428571428571, 0.7309999999999999, 0.6942857142857144, 0.3057142857142857, 0.269, 0.5866666666666667, 0.5671428571428572, 0.19657142857142856, 0.15933333333333333, 0.15933333333333333, 0.028329716039954534], 'Bernoulli Naive Bayes', 'Default with balancer'), ([0.30440505513543065, 0.26271930158486712, 0.6645714285714287, 0.796, 0.5331428571428571, 0.4668571428571429, 0.20400000000000001, 0.796, 0.5331428571428571, 0.4668571428571429, 0.20400000000000001, 0.20400000000000001, 0.018985024707270426], 'Decision Tree', 'Default with balancer'), ([0.28022137739927055, 0.24818710908872915, 0.6522857142857144, 0.753, 0.5515714285714285, 0.4484285714285715, 0.24699999999999997, 0.49000000000000005, 0.2978571428571429, 0.22400000000000003, 0.08366666666666665, 0.08366666666666665, 0.029133268332734892], 'Extreme Learning Machine', 'Default with balancer'), ([0.33967926018457484, 0.30975722668233141, 0.6851190476190477, 0.7646666666666666, 0.6055714285714285, 0.3944285714285714, 0.23533333333333334, 0.734, 0.5714285714285714, 0.36628571428571427, 0.20666666666666664, 0.20666666666666664, 0.007281958711541847], 'Gaussian Naive Bayes', 'Default with balancer'), ([0.36123565713265238, 0.34521331073970407, 0.6956428571428572, 0.714, 0.6772857142857143, 0.3227142857142857, 0.286, 0.5316666666666666, 0.491, 0.18442857142857141, 0.14866666666666667, 0.14866666666666667, 0.00717253432907512], 'K-nearest neighbours', 'Default with balancer'), ([0.40398354199026415, 0.3945312393387484, 0.7157619047619047, 0.7006666666666665, 0.7308571428571429, 0.26914285714285713, 0.2993333333333333, 0.5269999999999999, 0.5774285714285715, 0.15885714285714286, 0.174, 0.174, 0.02725729671968407], 'Logistic regression', 'Default with balancer'), ([0.35062910862847607, 0.31141344843372287, 0.6907380952380953, 0.8033333333333333, 0.5781428571428572, 0.4218571428571428, 0.19666666666666668, 0.6246666666666667, 0.3092857142857143, 0.24914285714285717, 0.06166666666666665, 0.06166666666666665, 0.24162397857999132], 'Random forest', 'Default with balancer'), ([0.40704984710095315, 0.39104806545942489, 0.7199285714285714, 0.739, 0.7008571428571428, 0.2991428571428571, 0.261, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11821343942718987], 'SVM (RDF)', 'Default with balancer'), ([0.40704984710095315, 0.39104806545942489, 0.7199285714285714, 0.739, 0.7008571428571428, 0.2991428571428571, 0.261, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12704261362778874], 'SVM (linear)', 'Default with balancer'), ([0.40704984710095315, 0.39104806545942489, 0.7199285714285714, 0.739, 0.7008571428571428, 0.2991428571428571, 0.261, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11644680163849844], 'SVM (polynomial)', 'Default with balancer')], 'Default with balancer'), ('Tuned', [([0.3962715479646407, 0.38641079460889394, 0.711857142857143, 0.6980000000000001, 0.7257142857142858, 0.2742857142857143, 0.30199999999999994, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24589672149457115], 'AdaBoost', 'Tuned'), ([0.40230016230737686, 0.38451544052877223, 0.7178571428571427, 0.7450000000000001, 0.6907142857142857, 0.3092857142857143, 0.255, 0.5750000000000002, 0.5094285714285713, 0.18614285714285717, 0.13399999999999998, 0.13399999999999998, 5.556227801943377], 'Artificial neural network', 'Tuned'), ([0.39386907874278759, 0.37732730322102309, 0.713095238095238, 0.7343333333333332, 0.6918571428571428, 0.30814285714285716, 0.26566666666666666, 0.5956666666666668, 0.5637142857142857, 0.20028571428571432, 0.15966666666666668, 0.15966666666666668, 0.01982310982585638], 'Bernoulli Naive Bayes', 'Tuned'), ([0.28471436426738977, 0.26533066473369316, 0.6548809523809523, 0.6913333333333334, 0.6184285714285715, 0.38157142857142856, 0.30866666666666664, 0.3663333333333333, 0.46514285714285714, 0.15185714285714286, 0.18999999999999997, 0.18999999999999997, 0.00837674588488877], 'Decision Tree', 'Tuned'), ([0.40904177798850699, 0.39871810527367224, 0.7188809523809523, 0.7093333333333335, 0.7284285714285715, 0.2715714285714285, 0.2906666666666667, 0.44300000000000006, 0.48442857142857154, 0.1104285714285714, 0.12166666666666667, 0.12166666666666667, 0.03958141023125635], 'Extreme Learning Machine', 'Tuned'), ([0.33967926018457484, 0.30975722668233141, 0.6851190476190477, 0.7646666666666666, 0.6055714285714285, 0.3944285714285714, 0.23533333333333334, 0.734, 0.5714285714285714, 0.36628571428571427, 0.20666666666666664, 0.20666666666666664, 0.006950274542001367], 'Gaussian Naive Bayes', 'Tuned'), ([0.37353424335391361, 0.3589147615949343, 0.7018095238095239, 0.7133333333333333, 0.6902857142857144, 0.3097142857142857, 0.2866666666666667, 0.36733333333333335, 0.3741428571428571, 0.10142857142857142, 0.06833333333333333, 0.06833333333333333, 0.006729886289192422], 'K-nearest neighbours', 'Tuned'), ([0.40529296506752105, 0.38663763563231207, 0.7195952380952382, 0.7503333333333334, 0.6888571428571428, 0.3111428571428571, 0.24966666666666665, 0.5900000000000001, 0.5234285714285714, 0.193, 0.14266666666666666, 0.14266666666666666, 0.02038028368837154], 'Logistic regression', 'Tuned'), ([0.38755348257621747, 0.38040595659915893, 0.7058095238095239, 0.6743333333333333, 0.7372857142857142, 0.2627142857142857, 0.32566666666666666, 0.35933333333333334, 0.47700000000000004, 0.08014285714285714, 0.11566666666666667, 0.11566666666666667, 2.4031617760256454], 'Random forest', 'Tuned'), ([0.40454784966078911, 0.38971311318876906, 0.7182619047619048, 0.7316666666666666, 0.7048571428571427, 0.2951428571428572, 0.2683333333333333, 0.22666666666666666, 0.6759999999999999, 0.04214285714285714, 0.2376666666666666, 0.2376666666666666, 1.4676357390598274], 'SVM (RDF)', 'Tuned'), ([0.39624588872515687, 0.38249238790617601, 0.7135238095238096, 0.7213333333333334, 0.7057142857142857, 0.2942857142857143, 0.2786666666666666, 0.15733333333333333, 0.667142857142857, 0.02485714285714286, 0.24800000000000005, 0.24800000000000005, 1.3587341787435965], 'SVM (linear)', 'Tuned'), ([0.28910213743449875, 0.27472423685959518, 0.6227142857142856, 0.35899999999999993, 0.8864285714285713, 0.11357142857142857, 0.641, 0.096, 0.35485714285714287, 0.023714285714285716, 0.11466666666666667, 0.11466666666666667, 2.4950097048107467], 'SVM (polynomial)', 'Tuned')], 'Tuned')], [('Default without balancer', [([0.6913011211167619, 0.69113549745655212, 0.8463592545254294, 0.8394341618191431, 0.8532843472317155, 0.14671565276828435, 0.1605658381808567, 0.01629825489159175, 0.0002631578947368421, 0.0, 0.0, 0.0, 1.127865726472249], 'AdaBoost', 'Default without balancer'), ([0.71337503276175818, 0.71329868793651219, 0.8569910118549101, 0.8452670544685352, 0.868714969241285, 0.13128503075871498, 0.15473294553146483, 0.7729984135378107, 0.8141421736158578, 0.08819548872180452, 0.10556319407720785, 0.10556319407720785, 4.280004769393775], 'Artificial neural network', 'Default without balancer'), ([0.71840804581835938, 0.71839377221685097, 0.8593072186379125, 0.8449814912744579, 0.8736329460013671, 0.12636705399863296, 0.15501850872554204, 0.8006557377049182, 0.8433902939166098, 0.09110731373889269, 0.11692226335272342, 0.11692226335272342, 0.02633831302316203], 'Bernoulli Naive Bayes', 'Default without balancer'), ([0.62435489512238895, 0.62414786025520586, 0.8119069038757175, 0.7879799048122687, 0.8358339029391662, 0.16416609706083393, 0.21202009518773135, 0.7879799048122687, 0.8358339029391662, 0.16416609706083393, 0.21202009518773135, 0.21202009518773135, 0.019189535104850785], 'Decision Tree', 'Default without balancer'), ([0.67600552340659392, 0.67541649483798538, 0.8383530984448766, 0.8283500793231096, 0.848356117566644, 0.15164388243335614, 0.17164992067689053, 0.5745637228979377, 0.6898496240601504, 0.06315447710184552, 0.07199894235854046, 0.07199894235854046, 0.1415998599367893], 'Extreme Learning Machine', 'Default without balancer'), ([0.54234117363858192, 0.50868468050311511, 0.7450910209637839, 0.5525436277102063, 0.9376384142173617, 0.062361585782638396, 0.4474563722897937, 0.529090428344791, 0.9313773069036226, 0.05949760765550238, 0.42594394500264404, 0.42594394500264404, 0.006548860628050513], 'Gaussian Naive Bayes', 'Default without balancer'), ([0.67860774266695512, 0.67823942063109721, 0.8380451904965529, 0.8052564780539397, 0.8708339029391661, 0.1291660970608339, 0.1947435219460603, 0.6567107350608143, 0.7648222829801778, 0.061609706083390295, 0.10392384981491273, 0.10392384981491273, 0.00928987349032102], 'K-nearest neighbours', 'Default without balancer'), ([0.71962150188730323, 0.71884759371243589, 0.8612225265341135, 0.8683712321523004, 0.8540738209159262, 0.14592617908407385, 0.13162876784769958, 0.7354362771020624, 0.7901127819548872, 0.06990088858509912, 0.07589635113696457, 0.07589635113696457, 0.021269353022679016], 'Logistic regression', 'Default without balancer'), ([0.72513121639711908, 0.72475352856944064, 0.8612241621553043, 0.8313273400317293, 0.891120984278879, 0.10887901572112098, 0.16867265996827077, 0.6647964040190375, 0.7470471633629528, 0.05272385509227615, 0.07360126916975146, 0.07360126916975146, 0.25564826022778464], 'Random forest', 'Default without balancer'), ([0.72275159680057999, 0.71347595175949741, 0.8631446227100851, 0.9283500793231095, 0.7979391660970608, 0.2020608339029392, 0.07164992067689051, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10718980074197237], 'SVM (RDF)', 'Default without balancer'), ([0.72275159680057999, 0.71347595175949741, 0.8631446227100851, 0.9283500793231095, 0.7979391660970608, 0.2020608339029392, 0.07164992067689051, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0840313753663827], 'SVM (linear)', 'Default without balancer'), ([0.72275159680057999, 0.71347595175949741, 0.8631446227100851, 0.9283500793231095, 0.7979391660970608, 0.2020608339029392, 0.07164992067689051, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0994677296176332], 'SVM (polynomial)', 'Default without balancer')], 'Default without balancer'), ('Default with balancer', [([0.69287704251370108, 0.69188303738837464, 0.8479705953263525, 0.8567477525118985, 0.8391934381408065, 0.16080656185919343, 0.14325224748810156, 0.017620306716023267, 0.0010492139439507863, 0.0, 0.0, 0.0, 1.1190610633333233], 'AdaBoost', 'Default with balancer'), ([0.72582977632062096, 0.72481732405392219, 0.8644169191186226, 0.8739978847170808, 0.8548359535201641, 0.14516404647983597, 0.12600211528291907, 0.7779164463246959, 0.7919036226930964, 0.08823991797676009, 0.0794711792702274, 0.0794711792702274, 3.831571303946222], 'Artificial neural network', 'Default with balancer'), ([0.72446831519682697, 0.72276809848600387, 0.864145750294683, 0.8830830248545745, 0.8452084757347915, 0.15479152426520848, 0.11691697514542569, 0.8394077207826547, 0.8063226247436773, 0.12584757347915243, 0.0911951348492861, 0.0911951348492861, 0.009162005439524279], 'Bernoulli Naive Bayes', 'Default with balancer'), ([0.68946634782760952, 0.67343186491860807, 0.8448480715755062, 0.9361554732945532, 0.7535406698564592, 0.24645933014354063, 0.06384452670544685, 0.9361554732945532, 0.7535406698564592, 0.24645933014354063, 0.06384452670544685, 0.06384452670544685, 0.011674449633087532], 'Decision Tree', 'Default with balancer'), ([0.68311166774138354, 0.675730503166734, 0.8433329161445029, 0.8983712321523003, 0.7882946001367054, 0.2117053998632946, 0.10162876784769961, 0.7511263881544157, 0.6292788790157212, 0.13103896103896104, 0.05833421470121629, 0.05833421470121629, 0.02003042752483578], 'Extreme Learning Machine', 'Default with balancer'), ([0.70026045715225094, 0.68791223253842682, 0.8511846731992716, 0.9289687995769433, 0.7734005468215994, 0.22659945317840052, 0.07103120042305659, 0.9283183500793232, 0.7728742310321257, 0.22607997265892, 0.07103120042305659, 0.07103120042305659, 0.005760371167506939], 'Gaussian Naive Bayes', 'Default with balancer'), ([0.70903824711434371, 0.70464849614136116, 0.8566940092527362, 0.8960708619777897, 0.8173171565276828, 0.18268284347231717, 0.10392913802221046, 0.8198519301956637, 0.7407894736842104, 0.12291866028708134, 0.06806451612903226, 0.06806451612903226, 0.006489846926609211], 'K-nearest neighbours', 'Default with balancer'), ([0.72065670762621303, 0.71823059088715036, 0.8623546908712095, 0.8875938656795347, 0.8371155160628845, 0.16288448393711552, 0.11240613432046538, 0.7673611845584347, 0.7835885167464115, 0.08896787423103213, 0.07231094658910629, 0.07231094658910629, 0.02078559160103755], 'Logistic regression', 'Default with balancer'), ([0.73053271794067798, 0.73036166670752167, 0.8656059280695368, 0.8557641459545213, 0.8754477101845521, 0.1245522898154477, 0.14423585404547856, 0.7202221047065044, 0.7292617908407383, 0.06448735475051263, 0.06481755684822844, 0.06481755684822844, 0.2569766873239779], 'Random forest', 'Default with balancer'), ([0.72275159680057999, 0.71347595175949741, 0.8631446227100851, 0.9283500793231095, 0.7979391660970608, 0.2020608339029392, 0.07164992067689051, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08481412947998512], 'SVM (RDF)', 'Default with balancer'), ([0.72275159680057999, 0.71347595175949741, 0.8631446227100851, 0.9283500793231095, 0.7979391660970608, 0.2020608339029392, 0.07164992067689051, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08106212588180901], 'SVM (linear)', 'Default with balancer'), ([0.72275159680057999, 0.71347595175949741, 0.8631446227100851, 0.9283500793231095, 0.7979391660970608, 0.2020608339029392, 0.07164992067689051, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08364550726125097], 'SVM (polynomial)', 'Default with balancer')], 'Default with balancer'), ('Tuned', [([0.7382699399394056, 0.73571920104449995, 0.8713005556051563, 0.8984452670544686, 0.8441558441558442, 0.15584415584415584, 0.10155473294553145, 0.6267636171337916, 0.646244019138756, 0.04646958304853042, 0.07001057641459545, 0.07001057641459545, 0.219234391555154], 'AdaBoost', 'Tuned'), ([0.72162421747310845, 0.72083977861871029, 0.8621701819569836, 0.8684346906398732, 0.8559056732740944, 0.1440943267259057, 0.13156530936012695, 0.759576943416182, 0.8005570745044428, 0.0774982911825017, 0.08208355367530407, 0.08208355367530407, 8.404495546050313], 'Artificial neural network', 'Tuned'), ([0.72289549175536039, 0.72126642092808768, 0.863327013449686, 0.881448968799577, 0.845205058099795, 0.15479494190020504, 0.11855103120042307, 0.8384241142252776, 0.8073684210526316, 0.12506151742993848, 0.09249074563722899, 0.09249074563722899, 0.021561191623980135], 'Bernoulli Naive Bayes', 'Tuned'), ([0.69025026950953627, 0.68513596222886075, 0.847138116552378, 0.8893548387096775, 0.8049213943950786, 0.1950786056049214, 0.11064516129032258, 0.8613167636171338, 0.7339507860560494, 0.17183185235816814, 0.07066631411951349, 0.07066631411951349, 0.0062177198071729], 'Decision Tree', 'Tuned'), ([0.70600389536911057, 0.70288008893389298, 0.855074189427706, 0.885018508725542, 0.8251298701298702, 0.1748701298701299, 0.11498149127445798, 0.6980539397144367, 0.7266712235133287, 0.07775803144224197, 0.06709148598625067, 0.06709148598625067, 0.04782723927565495], 'Extreme Learning Machine', 'Tuned'), ([0.70026045715225094, 0.68791223253842682, 0.8511846731992716, 0.9289687995769433, 0.7734005468215994, 0.22659945317840052, 0.07103120042305659, 0.9283183500793232, 0.7728742310321257, 0.22607997265892, 0.07103120042305659, 0.07103120042305659, 0.005697221979056621], 'Gaussian Naive Bayes', 'Tuned'), ([0.73023962715530633, 0.73006762289860871, 0.8658638402289075, 0.8609518773135907, 0.8707758031442243, 0.12922419685577582, 0.13904812268640931, 0.706890534108937, 0.7710628844839371, 0.049572795625427206, 0.0749074563722898, 0.0749074563722898, 0.0052447936633477354], 'K-nearest neighbours', 'Tuned'), ([0.7175096983913285, 0.71427497227031667, 0.8608882299614716, 0.8924907456372289, 0.8292857142857144, 0.17071428571428574, 0.10750925436277103, 0.7934849286092016, 0.7783868762816131, 0.10988380041011621, 0.0697355896351137, 0.0697355896351137, 0.024731570670038835], 'Logistic regression', 'Tuned'), ([0.71725490689130811, 0.71496097574648998, 0.860627958350759, 0.884690639873083, 0.8365652768284347, 0.1634347231715653, 0.11530936012691698, 0.7100846113167636, 0.7903554340396446, 0.04852358168147642, 0.06481226864093073, 0.06481226864093073, 2.6965128185004184], 'Random forest', 'Tuned'), ([0.71830390717030257, 0.71432064019041031, 0.8613348232245919, 0.8983395029085142, 0.82433014354067, 0.17566985645933014, 0.10166049709148599, 0.8133209941829719, 0.7916883116883117, 0.11274777853725222, 0.07753040719196193, 0.07753040719196193, 0.5091415091207099], 'SVM (RDF)', 'Tuned'), ([0.72200002564901788, 0.71149849020157185, 0.8625435472846339, 0.9329032258064517, 0.7921838687628162, 0.20781613123718387, 0.0670967741935484, 0.9185510312004231, 0.7919241285030759, 0.20390635680109362, 0.0670967741935484, 0.0670967741935484, 0.6616800340498484], 'SVM (linear)', 'Tuned'), ([0.73548103073181292, 0.73464991505356281, 0.8692345663326625, 0.8778847170809095, 0.8605844155844157, 0.13941558441558444, 0.12211528291909043, 0.8022897937599154, 0.7721223513328777, 0.09369788106630213, 0.07359598096245372, 0.07359598096245372, 0.625512966411998], 'SVM (polynomial)', 'Tuned')], 'Tuned')]]
    parameter = "Balanced Accuracy"
    x_label = "Parameter tuning approach"
    difference_from = "using default parameters"
    datasets = ["Lima TB", "Test", "blash"]
    name_suffix = ""
    patterns = (None, "////")

    colors = ["#64B3DE", "#1f78b4", "#B9B914", "#FBAC44", "#bc1659", "#33a02c", "#6ABF20", "#ff7f00", "#6a3d9a", "#5a2add", "#b15928", "#e31a1c", "grey"]
    classifier_arr = []
    for i in range(len(results)):
        classifier_arr.append(list())
    index = 0
    for results_per_classifier in results:
        no_feature_selection = results[index][0][1]
        for i in range(len(no_feature_selection) + 1):
            classifier_arr[index].append(list())
        for i in range(1, len(results_per_classifier)):
            data_balancer_results = results_per_classifier[i][1]
            x = 0
            mean_classification = 0
            for (result_arr, data_balancer_name, _) in data_balancer_results:
                value = result_arr[2] - no_feature_selection[x][0][2]
                classifier_arr[index][x].append(value)
                mean_classification += value
                x += 1
            mean_classification /= float(len(data_balancer_results))
            classifier_arr[index][x].append(mean_classification)
        index += 1

    fig = plt.figure(figsize=(16, 5))

    classifiers = np.arange(len(classifier_arr[0]))
    data_balancers = np.arange(len(classifier_arr[0][0])) * 3
    bar_width = 0.2
    opacity = 0.9
    subplt_val = 101 + (10 * len(results))
    ax1 = plt.subplot(subplt_val)
    for i in range(len(classifier_arr[0])):
        if i + 1 != len(classifier_arr[0]):
            label = results[0][0][1][i][1]
        else:
            label = "Mean classification"
        plt.bar(data_balancers + (i * bar_width), classifier_arr[0][i], bar_width,
                alpha=opacity,
                color=colors[i],
                hatch=patterns[i % len(patterns)],
                label=label)

    legend = plt.legend(loc='lower center', bbox_to_anchor=(len(results) / 2.0, -0.31), fancybox=True, frameon=True, ncol=6)
    legend.get_frame().set_facecolor('#ffffff')

    plt.xlabel(x_label, x=len(results) / 2.0)
    plt.ylabel("Difference in {0} from {1}".format(parameter, difference_from))
    feature_selection_labels = [results[0][i][0] for i in range(1, len(results[0]))]
    plt.xticks(data_balancers + (bar_width / 2) * len(classifiers), feature_selection_labels)
    plt.title(datasets[0].replace("_", " "))

    for z in range(1, len(results)):
        ax2 = plt.subplot(subplt_val + z, sharey=ax1)
        color = iter(cm.Set1(np.linspace(0, 1, len(no_feature_selection) + 1)))
        for i in range(len(classifier_arr[z])):
            if i + 1 != len(classifier_arr[z]):
                label = results[z][0][1][i][1]
            else:
                label = "Mean classification"
            plt.bar(data_balancers + (i * bar_width), classifier_arr[z][i], bar_width,
                    alpha=opacity,
                    color=colors[i],
                    hatch=patterns[i % len(patterns)],
                    label=label)

        feature_selection_labels = [results[z][i][0] for i in range(1, len(results[z]))]
        plt.xticks(data_balancers + (bar_width / 2) * len(classifiers), feature_selection_labels)
        plt.title(datasets[z].replace("_", " "))

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    plt.locator_params(axis='y', nbins=15)
    name = "{3}_results_per_classifier_plot{0}_{4}_{1}_{2}".format(name_suffix, parameter, current_time, x_label, datasets)
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/{0}".format(name.replace(" ", "_")), bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)