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


def plot_percentage_difference_graph(results, datasets, name_suffix="", parameter="Balanced Accuracy", x_label="Feature selection approach", difference_from="no feature selection", figsize=(16, 5), legend_y=-0.31):
    print(results)
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

    fig = plt.figure(figsize=figsize)

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

    legend = plt.legend(loc='lower center', bbox_to_anchor=(len(results) / 2.0, legend_y), fancybox=True, frameon=True, ncol=6)
    legend.get_frame().set_facecolor('#ffffff')

    plt.xlabel(x_label, x=len(results) / 2.0)
    plt.ylabel("Difference in {0} from {1}".format(parameter, difference_from))
    feature_selection_labels = [results[0][i][0] for i in range(1, len(results[0]))]
    plt.xticks(data_balancers + (bar_width / 2) * len(classifiers), feature_selection_labels, rotation=12)
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
        plt.xticks(data_balancers + (bar_width / 2) * len(classifiers), feature_selection_labels, rotation=12)
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
    print(dataset_results)
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


def visualise_dataset_balancer_results(results, range=(-0.3, 0.3), colors=("#64B3DE", "#1f78b4", "#B9B914", "#FBAC44", "#bc1659", "#33a02c", "#6ABF20", "#ff7f00", "#6a3d9a", "grey", "#b15928", "#e31a1c"), exclude=("SVM (linear)", "SVM (polynomial)", "SVM (RDF)", "Logistic regression")):
    print(results)
    sns.set(style='ticks')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    markers = ["s", "d", "o", "^", "*"]
    hatches = [None, "////", ".."]

    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position(("axes", 0.5))

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

    ax.xaxis.set_label_coords(0.1, 0.53)
    ax.yaxis.set_label_coords(0.53, 0.9)

    plt.ylim(range[0], range[1])
    plt.xlim(range[0], range[1])
    balancer_labels = ([], [])
    classifier_labels = ([], [])
    data_set_index = 0
    for (data_set, dataset_result) in results:

        median_true_pos_per_classifier = {}
        median_true_neg_per_classifier = {}

        for (classifier_description, result_arr) in dataset_result:
            true_pos_arr = []
            true_neg_arr = []
            for (balancer_description, results) in result_arr:
                true_pos_arr.append(results[3])
                true_neg_arr.append(results[4])

            median_true_pos_per_classifier[classifier_description] = np.median(np.array(true_pos_arr))
            median_true_neg_per_classifier[classifier_description] = np.median(np.array(true_neg_arr))


        i = 0
        for (classifier_description, result_arr) in dataset_result:
            if classifier_description in exclude:
                continue
            balancer_index = 0
            for (balancer_description, results) in result_arr:
                if data_set_index == 0 and balancer_index == 0:
                    classifier_labels[0].append(mpatches.Patch(color=colors[i], label=classifier_description, alpha=0.8))
                    classifier_labels[1].append(classifier_description)
                ax.scatter(results[3] -  median_true_pos_per_classifier[classifier_description], results[4] - median_true_neg_per_classifier[classifier_description], marker=markers[balancer_index % len(markers)], hatch=hatches[balancer_index % len(hatches)], s=200, alpha=0.8, color=colors[i], edgecolor="black" if colors[i] != "black" else "grey", zorder=balancer_index % len(markers), lw=0.8)
                pt = ax.scatter(-99999999999, -9999999999, marker=markers[balancer_index % len(markers)], hatch=hatches[balancer_index % len(hatches)], s=200, alpha=0.8, color="white", edgecolor="black", zorder=data_set_index, lw=0.8)
                if i == 0:
                    balancer_labels[0].append(pt)
                    balancer_labels[1].append(balancer_description)
                balancer_index += 1
            i += 1
        data_set_index += 1
    legend = plt.legend(balancer_labels[0] + classifier_labels[0], balancer_labels[1] + classifier_labels[1], loc='lower center', bbox_to_anchor=(0.5, -0.2), fancybox=False, frameon=False, ncol=6)
    legend.get_frame().set_facecolor('#ffffff')

    sns.despine()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/classifier_dataset_plt_{0}.png".format(current_time), bbox_extra_artists=((legend,)), bbox_inches='tight')
    plt.close(fig)


def visualise_dataset_balancer_results_multi_dataset(dataset_results):
    print(dataset_results)
    sns.set(style='ticks')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    markers = ["s", "o", "^", "d", "*"]
    colors = ["#64B3DE", "#1f78b4", "#B9B914", "#FBAC44", "#bc1659", "#33a02c", "#6ABF20", "#ff7f00", "#6a3d9a", "grey", "#b15928", "#e31a1c"]
    hatches = [None, "////", ".."]

    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position(("axes", 0.5))

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

    ax.xaxis.set_label_coords(0.1, 0.53)
    ax.yaxis.set_label_coords(0.53, 0.9)

    plt.ylim(-0.3, 0.3)
    plt.xlim(-0.3, 0.3)
    data_set_labels = ([], [])
    balancer_labels = ([], [])
    data_set_index = 0
    for (data_set, dataset_result) in dataset_results:
        balancer_result_pos = {}
        balancer_result_neg = {}

        for (classifier_description, result_arr) in dataset_result:
            for (balancer_description, results) in result_arr:
                if balancer_description in balancer_result_pos:
                    balancer_result_pos[balancer_description] = balancer_result_pos[balancer_description] + results[3]
                    balancer_result_neg[balancer_description] = balancer_result_neg[balancer_description] + results[4]
                else:
                    balancer_result_pos[balancer_description] = results[3]
                    balancer_result_neg[balancer_description] = results[4]

        for (balancer_description, _) in dataset_result[0][1]:
            balancer_result_pos[balancer_description] = balancer_result_pos[balancer_description] / float(len(dataset_result))
            balancer_result_neg[balancer_description] = balancer_result_neg[balancer_description] / float(len(dataset_result))

        true_pos_arr = [value for _, value in balancer_result_pos.iteritems()]
        true_neg_arr = [value for _, value in balancer_result_neg.iteritems()]
        median_true_pos = np.median(np.array(true_pos_arr))
        median_true_neg = np.median(np.array(true_neg_arr))

        for (balancer_description, _) in dataset_result[0][1]:
            print(balancer_description, balancer_result_pos[balancer_description] - median_true_pos)
            print(balancer_description, balancer_result_neg[balancer_description] - median_true_neg)

        i = 0
        hatch_index = 0
        for key, value in balancer_result_pos.iteritems():
            if i != 0 and hatch_index == 0 and i % len(colors) == 0:
                hatch_index += 1

            if data_set_index == 0:
                balancer_labels[0].append(mpatches.Patch(facecolor=colors[i % len(colors)], hatch=hatches[hatch_index], label=key, alpha=0.8, edgecolor="black"))
                balancer_labels[1].append(key)

            ax.scatter(value - median_true_pos, balancer_result_neg[key] - median_true_neg, marker=markers[data_set_index % len(markers)], hatch=hatches[hatch_index], s=200, alpha=0.8, color=colors[i % len(colors)], edgecolor="black" if colors[i % len(colors)] != "black" else "grey", zorder=i % len(markers), lw=0.8)
            pt = ax.scatter(-99999999999, -9999999999, marker=markers[data_set_index % len(markers)], s=200, alpha=0.8, color="white", edgecolor="black", zorder=data_set_index, lw=0.8)
            if i == 0:
                data_set_labels[0].append(pt)
                data_set_labels[1].append(data_set)
            i += 1
            hatch_index = (hatch_index + 1) % len(hatches)
        data_set_index += 1
    legend = plt.legend(data_set_labels[0] + balancer_labels[0], data_set_labels[1] + balancer_labels[1], loc='lower center', bbox_to_anchor=(0.5, -0.2), fancybox=False, frameon=False, ncol=6)
    legend.get_frame().set_facecolor('#ffffff')

    sns.despine()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/classifier_dataset_plt_{0}.png".format(current_time), bbox_extra_artists=((legend,)), bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    results = [('Lima_TB', [('AdaBoost', [('None', [0.26353726173235603, 0.22775148524821276, 0.5819641928308753, 0.1863247863247863, 0.9776035993369643, 0.022396400663035755, 0.8136752136752137, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7839155827847515]), ('ClusterCentroids', [0.27374757468338362, 0.22611040765650936, 0.6994249912128196, 0.6712250712250711, 0.7276249112005683, 0.27237508879943173, 0.32877492877492875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2939192534134434]), ('EditedNearestNeighbours', [0.3264867513242774, 0.32609910841555112, 0.6692464815860506, 0.425071225071225, 0.9134217381008762, 0.08657826189912385, 0.5749287749287749, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9377565817753286]), ('InstanceHardnessThreshold', [0.15774637726911159, 0.10224143528817586, 0.6233645691448203, 0.7088319088319089, 0.5378972294577314, 0.46210277054226856, 0.29116809116809117, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8382336427608941]), ('NearMiss', [0.17163718907240466, 0.10819487122512295, 0.6347237856993956, 0.747008547008547, 0.5224390243902439, 0.4775609756097561, 0.252991452991453, 0.0, 0.0, 0.0, 0.0, 0.0, 1.9208520688339483]), ('NeighbourhoodCleaningRule', [0.32160243801155608, 0.32150333964709399, 0.6640586180818244, 0.41082621082621085, 0.9172910253374379, 0.08270897466256216, 0.5891737891737892, 0.0, 0.0, 0.0, 0.0, 0.0, 0.708050992760171]), ('OneSidedSelection', [0.20347151797193239, 0.1991460606162484, 0.5860057695174911, 0.23133903133903133, 0.9406725076959507, 0.05932749230404925, 0.7686609686609687, 0.059829059829059825, 0.6459957376272792, 0.024290788538953348, 0.3886039886039886, 0.3886039886039886, 2.3918972682794895]), ('RandomUnderSampler', [0.28796391784304293, 0.23334117356274642, 0.7122157254432888, 0.7085470085470085, 0.715884442339569, 0.28411555766043095, 0.2914529914529915, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2475893387555601]), ('TomekLinks', [0.22546229481737137, 0.22251415702465849, 0.5993085551716856, 0.2618233618233618, 0.9367937485200095, 0.06320625147999052, 0.7381766381766381, 0.06723646723646723, 0.661562869997632, 0.02429078853895335, 0.3883190883190883, 0.3883190883190883, 2.0777216383561345]), ('ADASYN', [0.32752722247603311, 0.28956263395991311, 0.7264331671696105, 0.664957264957265, 0.7879090693819559, 0.21209093061804402, 0.335042735042735, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0056792112012154]), ('RandomOverSampler', [0.018368544774985208, 0.00067457926893721432, 0.5014563106796117, 1.0, 0.0029126213592233006, 0.9970873786407767, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.826319796280103]), ('SMOTE', [0.29492652057881996, 0.29431899740517864, 0.6555707012136092, 0.4045584045584046, 0.9065829978688136, 0.09341700213118635, 0.5954415954415955, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5330851634154143]), ('SMOTEENN', [0.32099444787794562, 0.29672561651142759, 0.710452055727926, 0.598005698005698, 0.8228984134501539, 0.17710158654984606, 0.4019943019943019, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7954865152693245]), ('SMOTETomek', [0.29492652057881996, 0.29431899740517864, 0.6555707012136092, 0.4045584045584046, 0.9065829978688136, 0.09341700213118635, 0.5954415954415955, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4185859989664351])]), ('Artificial neural network', [('None', [0.24120174830895952, 0.2206980527201986, 0.5837563660675193, 0.20056980056980053, 0.966942931565238, 0.03305706843476202, 0.7994301994301993, 0.14871794871794872, 0.8910632251953586, 0.020426237272081458, 0.5584045584045584, 0.5584045584045584, 2.047962378470796]), ('ClusterCentroids', [0.30167564280829762, 0.2743487339991213, 0.7020666503403895, 0.5977207977207979, 0.8064125029599809, 0.19358749704001893, 0.4022792022792022, 0.24643874643874644, 0.145010655931802, 0.025285342173810087, 0.04444444444444444, 0.04444444444444444, 1.8810809880752828]), ('EditedNearestNeighbours', [0.36127755320434474, 0.36119453647881816, 0.6779475710614712, 0.42592592592592593, 0.9299692161970163, 0.07003078380298366, 0.5740740740740741, 0.3282051282051282, 0.8920104191333176, 0.038905991001657585, 0.48461538461538456, 0.48461538461538456, 4.126796676149127]), ('InstanceHardnessThreshold', [0.23338515280243469, 0.14790605537806789, 0.6826391494684179, 0.8282051282051281, 0.5370731707317074, 0.46292682926829265, 0.1717948717948718, 0.6045584045584046, 0.3036609045702108, 0.25484252900781434, 0.12706552706552707, 0.12706552706552707, 1.9414018541504632]), ('NearMiss', [0.17264010282736675, 0.12891083283415972, 0.6326814021438695, 0.6279202279202278, 0.6374425763675112, 0.3625574236324888, 0.37207977207977205, 0.4863247863247864, 0.1915747099218565, 0.1720672507695951, 0.0301994301994302, 0.0301994301994302, 1.0512686610415898]), ('NeighbourhoodCleaningRule', [0.34050084021444293, 0.33925684091629982, 0.6597583980818648, 0.3817663817663817, 0.9377504143973479, 0.06224958560265215, 0.6182336182336183, 0.2621082621082621, 0.8920198910726972, 0.028221643381482357, 0.5065527065527066, 0.5065527065527066, 2.0896173731822003]), ('OneSidedSelection', [0.28417219105569169, 0.27502123013465973, 0.6154727705355223, 0.2766381766381766, 0.9543073644328676, 0.045692635567132364, 0.7233618233618233, 0.22364672364672367, 0.9367748046412503, 0.039853184939616385, 0.6937321937321937, 0.6937321937321937, 0.7442178542277418]), ('RandomUnderSampler', [0.28388838640243147, 0.24916136601103522, 0.6974634429690079, 0.6196581196581197, 0.7752687662798957, 0.22473123372010423, 0.3803418803418803, 0.15754985754985756, 0.0, 0.01750414397347857, 0.0, 0.0, 0.6811521668783698]), ('TomekLinks', [0.33000101641414148, 0.3039947177286233, 0.6171485819413832, 0.26153846153846155, 0.972758702344305, 0.027241297655695003, 0.7384615384615385, 0.12706552706552707, 0.9338385034335781, 0.0107080274686242, 0.6037037037037037, 0.6037037037037037, 3.977270778897747]), ('ADASYN', [0.31753404280713404, 0.28304500560265144, 0.7173313620365479, 0.6418803418803419, 0.792782382192754, 0.207217617807246, 0.35811965811965807, 0.4330484330484331, 0.5369452995500829, 0.08657826189912385, 0.17094017094017094, 0.17094017094017094, 1.380340221396871]), ('RandomOverSampler', [0.28395982205337683, 0.24268158178100763, 0.7020722431023165, 0.6492877492877492, 0.7548567369168836, 0.2451432630831163, 0.35071225071225065, 0.5008547008547009, 0.5680606204120294, 0.11088799431683638, 0.1564102564102564, 0.1564102564102564, 5.278073064830053]), ('SMOTE', [0.29502815422322715, 0.25866760572398129, 0.7054916334575345, 0.6347578347578346, 0.776225432157234, 0.22377456784276584, 0.36524216524216524, 0.4709401709401709, 0.4960928250059199, 0.09436419606914516, 0.17122507122507122, 0.17122507122507122, 5.795810713888983]), ('SMOTEENN', [0.3211774813453297, 0.29993001878443148, 0.7071499919380573, 0.5826210826210826, 0.831678901255032, 0.168321098744968, 0.41737891737891736, 0.4188034188034188, 0.647771726260952, 0.0768458441865972, 0.23076923076923075, 0.23076923076923075, 7.093864858077069]), ('SMOTETomek', [0.31542630303611879, 0.2805918992848434, 0.7163486483259157, 0.6418803418803418, 0.7908169547714895, 0.20918304522851053, 0.35811965811965807, 0.4635327635327635, 0.4902012787118163, 0.09242244849632962, 0.14871794871794872, 0.14871794871794872, 1.8905950403044613])]), ('Bernoulli Naive Bayes', [('None', [0.36111520399200969, 0.3487114554715951, 0.7182341781844506, 0.5678062678062678, 0.8686620885626333, 0.13133791143736678, 0.43219373219373214, 0.5307692307692308, 0.8472602415344541, 0.1157613071276344, 0.4025641025641026, 0.4025641025641026, 0.0775290510941854]), ('ClusterCentroids', [0.11588369665990385, 0.040986792963574326, 0.5732207022593003, 0.9256410256410257, 0.2208003788775752, 0.7791996211224248, 0.07435897435897436, 0.8507122507122507, 0.14591522614255267, 0.6634051622069619, 0.014814814814814814, 0.014814814814814814, 0.011038449835547137]), ('EditedNearestNeighbours', [0.34681544528068348, 0.32819914260469274, 0.7186491017511619, 0.59002849002849, 0.8472697134738338, 0.15273028652616621, 0.40997150997151, 0.5678062678062678, 0.833653800615676, 0.13327492304049254, 0.4025641025641026, 0.4025641025641026, 0.04607303102164373]), ('InstanceHardnessThreshold', [0.26998022949918266, 0.21852043455768244, 0.699727931359473, 0.6874643874643874, 0.7119914752545584, 0.28800852474544164, 0.3125356125356125, 0.6276353276353277, 0.6477811982003315, 0.2306180440445181, 0.2749287749287749, 0.2749287749287749, 0.007702584895362463]), ('NearMiss', [0.23585224542479621, 0.1671190692673733, 0.682750316574612, 0.7535612535612536, 0.6119393795879706, 0.38806062041202943, 0.24643874643874644, 0.7313390313390313, 0.5711011129528771, 0.33260241534454177, 0.23903133903133905, 0.23903133903133905, 0.011281749730741808]), ('NeighbourhoodCleaningRule', [0.34143074384672367, 0.32338066364496243, 0.7149359261080785, 0.5826210826210827, 0.8472507695950746, 0.1527492304049254, 0.41737891737891736, 0.5603988603988603, 0.8385081695477149, 0.12939616386455127, 0.4025641025641026, 0.4025641025641026, 0.011883358528658405]), ('OneSidedSelection', [0.36259129108411753, 0.35039947823251261, 0.7187243510473442, 0.5678062678062678, 0.8696424342884205, 0.13035756571157944, 0.43219373219373214, 0.5307692307692308, 0.8433720104191332, 0.11576130712763437, 0.41737891737891736, 0.41737891737891736, 0.00978784007650546]), ('RandomUnderSampler', [0.30008347993439832, 0.26776272150099489, 0.7052588490784086, 0.6196581196581196, 0.7908595784986975, 0.2091404215013024, 0.3803418803418803, 0.6122507122507123, 0.7441487094482595, 0.1702344304996448, 0.3504273504273504, 0.3504273504273504, 0.021287835246727838]), ('TomekLinks', [0.36515190354283783, 0.35207580966688812, 0.7214524449949503, 0.5752136752136752, 0.8676912147762254, 0.13230878522377457, 0.4247863247863248, 0.5307692307692308, 0.8433767463888231, 0.11770779067013973, 0.4025641025641026, 0.4025641025641026, 0.025816050789909895]), ('ADASYN', [0.29288067978024795, 0.26106937359047211, 0.7007125138216779, 0.6125356125356125, 0.7888894151077432, 0.21111058489225668, 0.38746438746438744, 0.5974358974358974, 0.7314894624674402, 0.16245796826900308, 0.3133903133903134, 0.3133903133903134, 0.05230283652661427]), ('RandomOverSampler', [0.31131831861781001, 0.28394939023784194, 0.7077097336320637, 0.6051282051282051, 0.8102912621359224, 0.18970873786407766, 0.39487179487179486, 0.5974358974358974, 0.7567795406109401, 0.15954061094008998, 0.33561253561253557, 0.33561253561253557, 0.027162652319017155]), ('SMOTE', [0.31516881319868012, 0.28996516280178286, 0.7077636508254552, 0.5974358974358974, 0.818091404215013, 0.18190859578498697, 0.4025641025641026, 0.5974358974358974, 0.7703954534690978, 0.15175941273975846, 0.3504273504273504, 0.3504273504273504, 0.04042943940806154]), ('SMOTEENN', [0.34564833580809551, 0.32539187649171086, 0.7199232530040013, 0.5974358974358974, 0.8424106085721051, 0.15758939142789485, 0.4025641025641026, 0.59002849002849, 0.79862183282027, 0.14592469808193226, 0.3800569800569801, 0.3800569800569801, 0.032596752459561174]), ('SMOTETomek', [0.31516881319868012, 0.28996516280178286, 0.7077636508254552, 0.5974358974358974, 0.818091404215013, 0.18190859578498697, 0.4025641025641026, 0.5974358974358974, 0.7703954534690978, 0.15175941273975846, 0.3504273504273504, 0.3504273504273504, 0.04897632927471207])]), ('Decision Tree', [('None', [0.15456765121120949, 0.1520904450220788, 0.587987705368714, 0.31310541310541307, 0.8628699976320151, 0.13713000236798484, 0.6868945868945869, 0.2760683760683761, 0.8492493488041676, 0.09630120767227088, 0.6868945868945869, 0.6868945868945869, 0.014083924578240025]), ('ClusterCentroids', [-0.0051129387816406533, -0.0022865577570907458, 0.4966422514611006, 0.6945868945868946, 0.2986976083353066, 0.7013023916646934, 0.3054131054131054, 0.6945868945868946, 0.2928676296471703, 0.7013023916646934, 0.29772079772079774, 0.29772079772079774, 0.005935792976542675]), ('EditedNearestNeighbours', [0.21036102975756513, 0.19427372514904173, 0.6379273116355758, 0.4703703703703703, 0.8054842529007814, 0.19451574709921854, 0.5296296296296297, 0.45555555555555555, 0.7967369168837319, 0.18868103244139237, 0.5222222222222223, 0.5222222222222223, 0.009062468304604998]), ('InstanceHardnessThreshold', [0.04948830929007262, 0.028068924981981658, 0.5384659978290098, 0.6421652421652422, 0.4347667534927776, 0.5652332465072224, 0.3578347578347578, 0.6347578347578346, 0.42115084063461994, 0.5623064172389297, 0.3350427350427351, 0.3350427350427351, 0.005400895440207165]), ('NearMiss', [0.13568539538275209, 0.070524893700131042, 0.6030198688095918, 0.805982905982906, 0.40005683163627753, 0.5999431683637224, 0.194017094017094, 0.7834757834757834, 0.3084489699265925, 0.567838029836609, 0.08917378917378918, 0.08917378917378918, 0.00486328115566792]), ('NeighbourhoodCleaningRule', [0.17728263788782286, 0.16747720364741636, 0.6123964883661781, 0.4037037037037036, 0.8210892730286525, 0.17891072697134738, 0.5962962962962963, 0.39629629629629626, 0.8123372010419134, 0.1614207909069382, 0.5886039886039887, 0.5886039886039887, 0.009972880815654506]), ('OneSidedSelection', [0.1758046904697475, 0.17014575894163786, 0.605325321313955, 0.36524216524216524, 0.8454084773857448, 0.15459152261425527, 0.6347578347578346, 0.3282051282051282, 0.8317878285578972, 0.10889888704712289, 0.6347578347578346, 0.6347578347578346, 0.011697714067681275]), ('RandomUnderSampler', [0.22395436966236892, 0.17675127525620016, 0.668090776800225, 0.6504273504273504, 0.6857542031730998, 0.3142457968269003, 0.34957264957264955, 0.6504273504273504, 0.628415818138764, 0.3093772199857921, 0.3051282051282051, 0.3051282051282051, 0.005006966949853364]), ('TomekLinks', [0.15168562165517346, 0.14784804039432564, 0.589352757555457, 0.3284900284900285, 0.8502154866208856, 0.1497845133791144, 0.6715099715099715, 0.2914529914529915, 0.8375657115794459, 0.10895571868340041, 0.6715099715099715, 0.6715099715099715, 0.009687018531842284]), ('ADASYN', [0.14660597626896224, 0.14178202560764142, 0.5882139322513463, 0.33589743589743587, 0.840530428605257, 0.15946957139474308, 0.6641025641025641, 0.2766381766381766, 0.8113852711342646, 0.08556950035519773, 0.6566951566951567, 0.6566951566951567, 0.018977089964213434]), ('RandomOverSampler', [0.17308182746025194, 0.1705976729790164, 0.5978365051512103, 0.3279202279202279, 0.8677527823821928, 0.13224721761780725, 0.672079772079772, 0.29088319088319087, 0.8366374615202462, 0.10310206014681504, 0.672079772079772, 0.672079772079772, 0.014393332013199256]), ('SMOTE', [0.18320743621454208, 0.18177478258793955, 0.583043966934566, 0.239031339031339, 0.927056594837793, 0.07294340516220696, 0.760968660968661, 0.239031339031339, 0.8891214776225432, 0.07294340516220696, 0.7239316239316239, 0.7239316239316239, 0.017464766794333286]), ('SMOTEENN', [0.21392512514565695, 0.21351544557166102, 0.6020034770922764, 0.28376068376068375, 0.9202462704238693, 0.07975372957613071, 0.7162393162393161, 0.24643874643874644, 0.9202462704238693, 0.07293866919251717, 0.7162393162393161, 0.7162393162393161, 0.017814623591519307]), ('SMOTETomek', [0.24048670750275059, 0.24002616088947026, 0.6145732668678441, 0.305982905982906, 0.9231636277527825, 0.07683637224721762, 0.694017094017094, 0.305982905982906, 0.8900876154392613, 0.07683637224721762, 0.6344729344729345, 0.6344729344729345, 0.018220022796533897])]), ('Extreme Learning Machine', [('None', [0.2396451100163941, 0.22698103814997517, 0.5911823800339073, 0.22421652421652422, 0.9581482358512906, 0.04185176414870945, 0.7757834757834757, 0.12735042735042734, 0.8958844423395691, 0.021411318967558608, 0.6108262108262108, 0.6108262108262108, 0.13202188853846053]), ('ClusterCentroids', [0.28985528304049601, 0.2559922052614485, 0.7003902722373004, 0.6196581196581197, 0.7811224248164811, 0.21887757518351886, 0.3803418803418803, 0.29829059829059823, 0.3190528060620412, 0.06224958560265213, 0.14131054131054127, 0.14131054131054127, 0.006754439770293885]), ('EditedNearestNeighbours', [0.32488344883369902, 0.32483598599629315, 0.660266381429062, 0.3954415954415954, 0.9250911674165285, 0.07490883258347146, 0.6045584045584046, 0.25413105413105413, 0.8472460336253848, 0.03502249585602652, 0.4028490028490029, 0.4028490028490029, 0.05150954604948765]), ('InstanceHardnessThreshold', [0.14946248930078945, 0.094090715316020423, 0.6173298643367315, 0.717094017094017, 0.5175657115794459, 0.48243428842055414, 0.2829059829059829, 0.4777777777777777, 0.24129291972531375, 0.21500828794695712, 0.11965811965811965, 0.11965811965811965, 0.00661860235982914]), ('NearMiss', [0.16071635285882785, 0.10009420495067245, 0.625555656751489, 0.7384615384615385, 0.5126497750414397, 0.48735022495856023, 0.26153846153846155, 0.5823361823361823, 0.27048543689320387, 0.3044612834477859, 0.10398860398860399, 0.10398860398860399, 0.004980403189584592]), ('NeighbourhoodCleaningRule', [0.32030726483632027, 0.31870416985881578, 0.6484929969040791, 0.3592592592592593, 0.9377267345488989, 0.06227326545110111, 0.6407407407407406, 0.2168091168091168, 0.8589675586076249, 0.033066540374141605, 0.4700854700854701, 0.4700854700854701, 0.03523079077899638]), ('OneSidedSelection', [0.25043430676983175, 0.23929172641685892, 0.5979085499094972, 0.23960113960113957, 0.9562159602178546, 0.043784039782145394, 0.7603988603988603, 0.12706552706552707, 0.8920198910726972, 0.016542742126450388, 0.6108262108262108, 0.6108262108262108, 0.09243403735347044]), ('RandomUnderSampler', [0.28259604285713352, 0.25241556788992148, 0.6932885305203562, 0.5977207977207977, 0.7888562633199147, 0.21114373668008524, 0.4022792022792022, 0.43247863247863244, 0.40268055884442333, 0.11190149183045228, 0.1564102564102564, 0.1564102564102564, 0.021521777453649804]), ('TomekLinks', [0.25295963015014816, 0.24481378139027055, 0.6027368507690554, 0.25413105413105413, 0.9513426474070567, 0.048657352592943404, 0.7458689458689459, 0.10512820512820513, 0.8900449917120531, 0.0214160549372484, 0.5886039886039887, 0.5886039886039887, 0.05500298238581536]), ('ADASYN', [0.25325422722522817, 0.23211099999497475, 0.6676875514834312, 0.5299145299145298, 0.8054605730523324, 0.19453942694766752, 0.4700854700854701, 0.24586894586894587, 0.5068576841108217, 0.062282737390480694, 0.1868945868945869, 0.1868945868945869, 0.0333879299103188]), ('RandomOverSampler', [0.28157318140569193, 0.25124609115891205, 0.6924017910330956, 0.5968660968660968, 0.7879374852000947, 0.2120625147999053, 0.40313390313390307, 0.3811965811965812, 0.45128581577077903, 0.06712289841345015, 0.13390313390313388, 0.13390313390313388, 0.04639934266766943]), ('SMOTE', [0.30853945991854925, 0.27754429257195734, 0.7092847943664898, 0.61994301994302, 0.7986265687899596, 0.20137343121004023, 0.3800569800569801, 0.3589743589743589, 0.4230641723892966, 0.08364669666114136, 0.14871794871794872, 0.14871794871794872, 0.027365502851985468]), ('SMOTEENN', [0.33307262981000701, 0.31914447948098124, 0.7049896881343721, 0.552991452991453, 0.856987923277291, 0.143012076722709, 0.44700854700854703, 0.3515669515669516, 0.6303386218328203, 0.076850580156287, 0.24586894586894587, 0.24586894586894587, 0.024304029480944678]), ('SMOTETomek', [0.30853945991854925, 0.27754429257195734, 0.7092847943664898, 0.61994301994302, 0.7986265687899596, 0.20137343121004023, 0.3800569800569801, 0.3589743589743589, 0.4230641723892966, 0.08364669666114136, 0.14871794871794872, 0.14871794871794872, 0.023533378572224706])]), ('Gaussian Naive Bayes', [('None', [0.059411741114617285, 0.012724675118516049, 0.5256108759992255, 0.9626780626780626, 0.08854368932038834, 0.9114563106796117, 0.037321937321937323, 0.9626780626780626, 0.08854368932038834, 0.9095098271371063, 0.037321937321937323, 0.037321937321937323, 0.010454952692357827]), ('ClusterCentroids', [0.041021176642820956, 0.0077650116457421703, 0.5158808262715437, 0.9626780626780626, 0.06908358986502486, 0.9309164101349751, 0.037321937321937323, 0.9626780626780626, 0.06811271607861709, 0.9260620412029363, 0.029914529914529912, 0.029914529914529912, 0.004978290163197396]), ('EditedNearestNeighbours', [0.078252280782372716, 0.01889159498774895, 0.5372826732997228, 0.9626780626780626, 0.11188728392138289, 0.8881127160786171, 0.037321937321937323, 0.9626780626780626, 0.1011745204830689, 0.8822732654511011, 0.037321937321937323, 0.037321937321937323, 0.007141727320585822]), ('InstanceHardnessThreshold', [0.13622047365423173, 0.065481771909727593, 0.6008081642180624, 0.8441595441595441, 0.3574567842765807, 0.6425432157234193, 0.15584045584045583, 0.8293447293447294, 0.34093298602888944, 0.6279753729576131, 0.14843304843304844, 0.14843304843304844, 0.004604888214867486]), ('NearMiss', [0.078379134114092905, 0.068863224117741595, 0.554119645976146, 0.3649572649572649, 0.7432820269950271, 0.2567179730049728, 0.635042735042735, 0.3649572649572649, 0.7413355434525218, 0.2567179730049728, 0.6276353276353276, 0.6276353276353276, 0.004510405749361013]), ('NeighbourhoodCleaningRule', [0.076779740633851165, 0.018368403847778381, 0.53631416749816, 0.9626780626780626, 0.10995027231825716, 0.890049727681743, 0.037321937321937323, 0.9626780626780626, 0.09730996921619701, 0.8842102770542268, 0.037321937321937323, 0.037321937321937323, 0.006277197668184797]), ('OneSidedSelection', [0.063573325489085966, 0.013989687043238663, 0.5280475324046245, 0.9626780626780626, 0.09341700213118635, 0.9065829978688136, 0.037321937321937323, 0.9626780626780626, 0.09244612834477858, 0.8997868813639593, 0.037321937321937323, 0.037321937321937323, 0.006587812546788641]), ('RandomUnderSampler', [0.12754199762537183, 0.077405816436184827, 0.5998468568205721, 0.7096866096866096, 0.4900071039545346, 0.5099928960454653, 0.2903133903133903, 0.6948717948717948, 0.4676012313521193, 0.4680558844423396, 0.2601139601139601, 0.2601139601139601, 0.009625740766676216]), ('TomekLinks', [0.060255742291928573, 0.012976860820098635, 0.5260963128924294, 0.9626780626780626, 0.0895145631067961, 0.9104854368932038, 0.037321937321937323, 0.9626780626780626, 0.08854368932038834, 0.9085342173810087, 0.037321937321937323, 0.037321937321937323, 0.00798935276190349]), ('ADASYN', [0.22115648013988654, 0.16178399348363326, 0.670051387295053, 0.7085470085470085, 0.6315557660430973, 0.3684442339569026, 0.2914529914529915, 0.6937321937321936, 0.6091783092588207, 0.3596732180914042, 0.2914529914529915, 0.2914529914529915, 0.010755304299944868]), ('RandomOverSampler', [0.065951770829089304, 0.013809373111999701, 0.5278582690233177, 0.97008547008547, 0.08563106796116504, 0.9143689320388348, 0.029914529914529912, 0.97008547008547, 0.08466019417475727, 0.9133980582524271, 0.029914529914529912, 0.029914529914529912, 0.011231640819321598]), ('SMOTE', [0.083999254607113549, 0.021001681319576582, 0.5411803763544233, 0.9626780626780626, 0.11968269003078383, 0.8803173099692161, 0.037321937321937323, 0.9626780626780626, 0.11482358512905518, 0.8793417002131185, 0.037321937321937323, 0.037321937321937323, 0.009131594453485548]), ('SMOTEENN', [0.096872177694398431, 0.026126921000302783, 0.5504155172495215, 0.9626780626780626, 0.13815297182098035, 0.8618470281790197, 0.037321937321937323, 0.9626780626780626, 0.1332891309495619, 0.8521051385271134, 0.037321937321937323, 0.037321937321937323, 0.008660691430531386]), ('SMOTETomek', [0.083999254607113549, 0.021001681319576582, 0.5411803763544233, 0.9626780626780626, 0.11968269003078383, 0.8803173099692161, 0.037321937321937323, 0.9626780626780626, 0.11482358512905518, 0.8793417002131185, 0.037321937321937323, 0.037321937321937323, 0.009955372882750169])]), ('K-nearest neighbours', [('None', [0.16556072608372613, 0.14384955441723379, 0.5523114298108378, 0.13475783475783476, 0.9698650248638409, 0.03013497513615913, 0.8652421652421651, 0.07492877492877492, 0.9250911674165285, 0.021378167179730047, 0.7085470085470085, 0.7085470085470085, 0.013337422542490027]), ('ClusterCentroids', [0.11128810870731909, 0.095007701210438777, 0.5797075302592708, 0.4336182336182336, 0.7257968269003078, 0.27420317309969217, 0.5663817663817664, 0.20997150997150998, 0.4173194411555766, 0.06515273502249586, 0.2754985754985755, 0.2754985754985755, 0.0018661041589194127]), ('EditedNearestNeighbours', [0.25409486973848056, 0.25408594550149299, 0.6278778538096559, 0.3433048433048433, 0.9124508643144684, 0.0875491356855316, 0.6566951566951567, 0.17863247863247864, 0.8034904096613781, 0.03892493488041676, 0.49230769230769234, 0.49230769230769234, 0.009788745659239595]), ('InstanceHardnessThreshold', [0.099710594713279491, 0.064795779954785249, 0.5784281303106782, 0.6276353276353277, 0.5292209329860289, 0.4707790670139711, 0.37236467236467236, 0.49401709401709404, 0.34635567132370354, 0.26946246744020835, 0.17122507122507122, 0.17122507122507122, 0.0035861076363570277]), ('NearMiss', [0.1908461888173705, 0.10667774060182966, 0.6472049750619489, 0.843019943019943, 0.45139000710395455, 0.5486099928960455, 0.15698005698005696, 0.6792022792022792, 0.270367037650959, 0.39099218565001187, 0.05242165242165242, 0.05242165242165242, 0.003197612642416914]), ('NeighbourhoodCleaningRule', [0.20886321495867649, 0.20692389693455704, 0.615547932128562, 0.35071225071225076, 0.8803836135448734, 0.11961638645512669, 0.6492877492877492, 0.1866096866096866, 0.8229836609045702, 0.08458441865972058, 0.5293447293447293, 0.5293447293447293, 0.008985493772005526]), ('OneSidedSelection', [0.19777852856441144, 0.18688089569057964, 0.57478774827579, 0.1943019943019943, 0.9552735022495856, 0.044726497750414394, 0.8056980056980058, 0.08974358974358973, 0.868690504380772, 0.021378167179730047, 0.6116809116809117, 0.6116809116809117, 0.009678868287220865]), ('RandomUnderSampler', [0.20591820693744509, 0.14663212781156743, 0.6596381031024648, 0.7094017094017094, 0.6098744968032205, 0.3901255031967795, 0.2905982905982906, 0.5678062678062678, 0.4221501302391665, 0.23248401610229696, 0.12706552706552707, 0.12706552706552707, 0.003039739385364726]), ('TomekLinks', [0.16556072608372613, 0.14384955441723379, 0.5521666116835429, 0.13447293447293446, 0.9698602888941512, 0.030139711105848926, 0.8655270655270655, 0.06723646723646723, 0.9144068197963533, 0.020407293393322277, 0.7082621082621083, 0.7082621082621083, 0.0024586571294733517]), ('ADASYN', [0.20189536578045589, 0.20188808838358008, 0.6003673749707376, 0.29116809116809117, 0.9095666587733838, 0.09043334122661614, 0.7088319088319087, 0.194017094017094, 0.8589817665166942, 0.06418186123608809, 0.5888888888888888, 0.5888888888888888, 0.029551277716866053]), ('RandomOverSampler', [0.16142856290770821, 0.15690938934747956, 0.596074191461357, 0.3438746438746439, 0.84827373904807, 0.15172626095192993, 0.6561253561253562, 0.26153846153846155, 0.7889367748046412, 0.0778025100639356, 0.5441595441595442, 0.5441595441595442, 0.050496198967394434]), ('SMOTE', [0.22647030401303292, 0.200840265366657, 0.6558146508774024, 0.5373219373219372, 0.7743073644328676, 0.22569263556713234, 0.46267806267806266, 0.36581196581196573, 0.651678901255032, 0.14981292919725314, 0.3504273504273504, 0.3504273504273504, 0.03605034315548039]), ('SMOTEENN', [0.23391700122183248, 0.21683959365171179, 0.6524246410748897, 0.4925925925925926, 0.8122566895571868, 0.18774331044281317, 0.5074074074074074, 0.3433048433048433, 0.7090977977740943, 0.1255031967795406, 0.39458689458689455, 0.39458689458689455, 0.025026382643723366]), ('SMOTETomek', [0.22647030401303292, 0.200840265366657, 0.6558146508774024, 0.5373219373219372, 0.7743073644328676, 0.22569263556713234, 0.46267806267806266, 0.36581196581196573, 0.651678901255032, 0.14981292919725314, 0.3504273504273504, 0.3504273504273504, 0.03188315326324087])]), ('Logistic regression', [('None', [0.30132091219095708, 0.26085714413252215, 0.7122361805146555, 0.656980056980057, 0.7674923040492541, 0.23250769595074594, 0.34301994301994304, 0.4635327635327634, 0.5223348330570684, 0.09339805825242718, 0.14131054131054127, 0.14131054131054127, 0.025942530512101068]), ('ClusterCentroids', [0.28699944069666905, 0.25110110087313764, 0.7002033768408383, 0.627065527065527, 0.7733412266161496, 0.22665877338385032, 0.3729344729344729, 0.3660968660968661, 0.3968884679138054, 0.051565237982476905, 0.1341880341880342, 0.1341880341880342, 0.4197131476124014]), ('EditedNearestNeighbours', [0.29416044980669787, 0.24568839153341659, 0.713260593696303, 0.6871794871794872, 0.7393417002131185, 0.2606582997868814, 0.3128205128205128, 0.5156695156695157, 0.5662183282026996, 0.12547004499171205, 0.19344729344729344, 0.19344729344729344, 0.3217366420509151]), ('InstanceHardnessThreshold', [0.17027615741272695, 0.10576224679202662, 0.633388046601402, 0.7541310541310541, 0.51264503907175, 0.48735496092825004, 0.24586894586894587, 0.6641025641025641, 0.3220601468150604, 0.3288609992896045, 0.13447293447293446, 0.13447293447293446, 0.0774188718612514]), ('NearMiss', [0.15392527228353958, 0.090987219039224465, 0.6198449745762084, 0.760968660968661, 0.4787212881837556, 0.5212787118162444, 0.239031339031339, 0.649002849002849, 0.29287710158654984, 0.3326071513142316, 0.09686609686609686, 0.09686609686609686, 0.007814273432856567]), ('NeighbourhoodCleaningRule', [0.28838481792720255, 0.2443306447805017, 0.7068448254808662, 0.6646723646723647, 0.7490172862893678, 0.25098271371063224, 0.33532763532763527, 0.5376068376068375, 0.5758702344304997, 0.12840634619938432, 0.149002849002849, 0.149002849002849, 0.31025777714453895]), ('OneSidedSelection', [0.30108937620355575, 0.26398181643563734, 0.7098374321059615, 0.6424501424501424, 0.7772247217617807, 0.22277527823821924, 0.3575498575498576, 0.4709401709401709, 0.5252758702344305, 0.0953445417949325, 0.1564102564102564, 0.1564102564102564, 2.1497462557172433]), ('RandomUnderSampler', [0.29675601128285639, 0.25389184939285414, 0.711194813640942, 0.6646723646723647, 0.7577172626095192, 0.2422827373904807, 0.3353276353276353, 0.4854700854700854, 0.4853374378403978, 0.11188728392138289, 0.1564102564102564, 0.1564102564102564, 0.005652043719123867]), ('TomekLinks', [0.30035125244743638, 0.25974072259099212, 0.7117483756366068, 0.656980056980057, 0.7665166942931565, 0.2334833057068435, 0.34301994301994304, 0.4931623931623932, 0.5310963769831873, 0.09436893203883495, 0.14131054131054127, 0.14131054131054127, 0.6191882718583521]), ('ADASYN', [0.3071505981838844, 0.26929602714729339, 0.714028940687714, 0.6498575498575498, 0.7782003315178783, 0.22179966848212174, 0.3501424501424501, 0.4333333333333333, 0.5028936774804641, 0.0933885863130476, 0.1561253561253561, 0.1561253561253561, 0.04723489367251332]), ('RandomOverSampler', [0.29126420237217993, 0.25268790699550481, 0.7046863229647979, 0.6418803418803418, 0.7674923040492541, 0.23250769595074594, 0.35811965811965807, 0.4709401709401709, 0.5349704001894388, 0.08852474544162917, 0.1564102564102564, 0.1564102564102564, 0.4377846548401578]), ('SMOTE', [0.28363206045859279, 0.24397344566727697, 0.7009287020677027, 0.6421652421652422, 0.7596921619701634, 0.2403078380298366, 0.3578347578347578, 0.4632478632478632, 0.5310869050438077, 0.09730049727681742, 0.17863247863247864, 0.17863247863247864, 0.5330048684127959]), ('SMOTEENN', [0.32440434865907353, 0.28591098953905303, 0.7246872202354088, 0.6643874643874644, 0.784986976083353, 0.21501302391664692, 0.33561253561253557, 0.48575498575498577, 0.601103480937722, 0.09437366800852473, 0.19373219373219372, 0.19373219373219372, 0.03805952938676782]), ('SMOTETomek', [0.28363206045859279, 0.24397344566727697, 0.7009287020677027, 0.6421652421652422, 0.7596921619701634, 0.2403078380298366, 0.3578347578347578, 0.4632478632478632, 0.5310869050438077, 0.09730049727681742, 0.17863247863247864, 0.17863247863247864, 0.4821434177416961])]), ('Random forest', [('None', [0.096888841653649027, 0.09653486411004264, 0.5452034206924096, 0.1789173789173789, 0.9114894624674402, 0.08851053753255979, 0.8210826210826211, 0.06666666666666667, 0.8210134975136159, 0.024314468387402317, 0.611965811965812, 0.611965811965812, 0.3479792223096948]), ('ClusterCentroids', [0.10714308769714218, 0.043481258969706671, 0.573168633578295, 0.872934472934473, 0.273402794222117, 0.7265972057778831, 0.12706552706552707, 0.61994301994302, 0.04964243428842055, 0.45906227800142074, 0.0150997150997151, 0.0150997150997151, 0.33545954097822284]), ('EditedNearestNeighbours', [0.22520888553792121, 0.22197657819270566, 0.6275831037872242, 0.38062678062678057, 0.8745394269476675, 0.12546057305233246, 0.6193732193732193, 0.17150997150997152, 0.7578403978214538, 0.05349751361591285, 0.4253561253561253, 0.4253561253561253, 0.35276613265458856]), ('InstanceHardnessThreshold', [0.12689300612566304, 0.07312051653794549, 0.5984970649806074, 0.7387464387464387, 0.45824769121477626, 0.5417523087852237, 0.26125356125356125, 0.5307692307692308, 0.25492777646223064, 0.36183755623964003, 0.15669515669515668, 0.15669515669515668, 0.2311771609618969]), ('NearMiss', [0.14379523150055337, 0.067130102277578763, 0.605034430229789, 0.8655270655270655, 0.3445417949325124, 0.6554582050674875, 0.13447293447293446, 0.805982905982906, 0.20923987686478807, 0.49681269239876863, 0.05213675213675213, 0.05213675213675213, 0.30434945351100673]), ('NeighbourhoodCleaningRule', [0.24209083134234163, 0.24057326742527285, 0.6322375028081871, 0.37435897435897436, 0.8901160312573999, 0.10988396874260005, 0.6256410256410256, 0.18717948717948718, 0.7743499881600757, 0.050603836135448735, 0.4700854700854701, 0.4700854700854701, 0.29146995397224273]), ('OneSidedSelection', [0.19252578694908198, 0.19102026527056171, 0.5869600471707977, 0.24586894586894587, 0.9280511484726498, 0.07194885152735023, 0.7541310541310541, 0.11965811965811965, 0.8172057778830215, 0.023348330570684347, 0.5301994301994302, 0.5301994301994302, 0.3635226443981878]), ('RandomUnderSampler', [0.24008664432256088, 0.18926410037809993, 0.6797572511946179, 0.6717948717948717, 0.6877196305943641, 0.3122803694056358, 0.3282051282051282, 0.4621082621082621, 0.40174757281553397, 0.1517452048306891, 0.11937321937321936, 0.11937321937321936, 0.3197386246733842]), ('TomekLinks', [0.12439339884840785, 0.12307195609185195, 0.5555169122017334, 0.1868945868945869, 0.92413923750888, 0.07586076249112006, 0.8131054131054132, 0.059829059829059825, 0.8297797774094245, 0.023348330570684347, 0.603988603988604, 0.603988603988604, 0.33308842351327783]), ('ADASYN', [0.13571365143517591, 0.13524626373701065, 0.5724291881455036, 0.2615384615384615, 0.8833199147525456, 0.11668008524745443, 0.7384615384615385, 0.09686609686609686, 0.8025621596021786, 0.035003551977267344, 0.5373219373219372, 0.5373219373219372, 0.38001723022086864]), ('RandomOverSampler', [0.19770912756337261, 0.19614040379533515, 0.6086306975840483, 0.33589743589743587, 0.8813639592706606, 0.11863604072933934, 0.6641025641025641, 0.17207977207977207, 0.787975372957613, 0.06516220696187544, 0.45527065527065524, 0.45527065527065524, 0.3767263425565943]), ('SMOTE', [0.15943177990603433, 0.15544429048683672, 0.5665045372883404, 0.1943019943019943, 0.9387070802746862, 0.061292919725313755, 0.8056980056980058, 0.08205128205128205, 0.8579730049727681, 0.0330712763438314, 0.5666666666666667, 0.5666666666666667, 0.42184338006945055]), ('SMOTEENN', [0.22359630227922342, 0.21449715906580535, 0.5882768626292187, 0.22421652421652422, 0.9523372010419134, 0.047662798958086665, 0.7757834757834757, 0.11965811965811965, 0.8978403978214539, 0.02041676533270187, 0.5669515669515669, 0.5669515669515669, 0.3585202053620158]), ('SMOTETomek', [0.15943177990603433, 0.15544429048683672, 0.5665045372883404, 0.1943019943019943, 0.9387070802746862, 0.061292919725313755, 0.8056980056980058, 0.08205128205128205, 0.8579730049727681, 0.0330712763438314, 0.5666666666666667, 0.5666666666666667, 0.3409458630565716])]), ('SVM (RDF)', [('None', [0.32485442821228006, 0.31102050957715932, 0.7003198466139502, 0.5455840455840455, 0.8550556476438551, 0.14494435235614495, 0.4544159544159544, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3501930702394169]), ('ClusterCentroids', [0.33315851875674424, 0.32171503911808608, 0.7014728798271304, 0.5381766381766381, 0.8647691214776225, 0.13523087852237745, 0.4618233618233618, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02366317876443702]), ('EditedNearestNeighbours', [0.33030939922946623, 0.31725042783799207, 0.7022639621716107, 0.5455840455840455, 0.858943878759176, 0.14105612124082406, 0.4544159544159544, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2512524209245157]), ('InstanceHardnessThreshold', [0.25162766939155273, 0.24905323837712778, 0.6408633429874254, 0.39743589743589747, 0.8842907885389533, 0.11570921146104665, 0.6025641025641025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.022370308477697165]), ('NearMiss', [0.30272852599460914, 0.28295816543954766, 0.6950822149496078, 0.5603988603988603, 0.8297655695003552, 0.17023443049964482, 0.43960113960113956, 0.0, 0.0, 0.0, 0.0, 0.0, 0.021305343179626846]), ('NeighbourhoodCleaningRule', [0.33169166905060588, 0.31882781536490645, 0.7027493990648147, 0.5455840455840455, 0.8599147525455837, 0.1400852474544163, 0.4544159544159544, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2840544388043469]), ('OneSidedSelection', [0.2986559262962763, 0.27240018685458134, 0.6991842663261086, 0.59002849002849, 0.8083400426237273, 0.19165995737627278, 0.40997150997151, 0.0, 0.0, 0.0, 0.0, 0.0, 0.26816146177957023]), ('RandomUnderSampler', [0.34132193818287604, 0.32730967002281575, 0.7096690015941733, 0.5603988603988603, 0.8589391427894861, 0.14106085721051387, 0.43960113960113956, 0.0, 0.0, 0.0, 0.0, 0.0, 0.019788190235175307]), ('TomekLinks', [0.32620722384556905, 0.31256616527306802, 0.7008052835071543, 0.5455840455840455, 0.8560265214302628, 0.14397347856973716, 0.4544159544159544, 0.0, 0.0, 0.0, 0.0, 0.0, 0.34451416089910225]), ('ADASYN', [0.33458284328515814, 0.32333278982553393, 0.7019583167203344, 0.5381766381766381, 0.8657399952640302, 0.1342600047359697, 0.4618233618233618, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9512980622944269]), ('RandomOverSampler', [0.32041473977345314, 0.29451049835567922, 0.7114649865443141, 0.6048433048433048, 0.8180866682453232, 0.18191333175467678, 0.3951566951566951, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7332612086993961]), ('SMOTE', [0.32893462545699581, 0.31568106425714249, 0.701776157293562, 0.5455840455840455, 0.8579682690030783, 0.14203173099692162, 0.4544159544159544, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8713456717974921]), ('SMOTEENN', [0.32756726184618445, 0.31411966355267484, 0.7012907204003581, 0.5455840455840455, 0.8569973952166705, 0.1430026047833294, 0.4544159544159544, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7266570956633416]), ('SMOTETomek', [0.32893462545699581, 0.31568106425714249, 0.701776157293562, 0.5455840455840455, 0.8579682690030783, 0.14203173099692162, 0.4544159544159544, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7681841013479911])]), ('SVM (linear)', [('None', [0.33303746431191034, 0.31072242228264868, 0.7150309963144441, 0.5974358974358974, 0.8326260951929907, 0.16737390480700925, 0.4025641025641026, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24589921950842553]), ('ClusterCentroids', [0.30789149145857464, 0.29161449761890434, 0.6938406690265557, 0.5452991452991454, 0.8423821927539663, 0.15761780724603364, 0.4547008547008547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02482383397166954]), ('EditedNearestNeighbours', [0.26761322183153385, 0.23068849613172071, 0.6894603962967685, 0.6202279202279202, 0.7586928723656168, 0.24130712763438314, 0.3797720797720797, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1973346285178863]), ('InstanceHardnessThreshold', [0.16958099771648996, 0.10391044283445938, 0.6326920344632871, 0.7615384615384615, 0.5038456073881127, 0.4961543926118873, 0.23846153846153845, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01330542528580203]), ('NearMiss', [0.15336639202989269, 0.082340233656525763, 0.6174079201334707, 0.8133903133903134, 0.421425526876628, 0.578574473123372, 0.1866096866096866, 0.0, 0.0, 0.0, 0.0, 0.0, 0.012744265850113123]), ('NeighbourhoodCleaningRule', [0.29176286507280191, 0.24988949333239974, 0.7075076858311526, 0.6572649572649573, 0.7577504143973478, 0.24224958560265214, 0.34273504273504274, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16957761206284516]), ('OneSidedSelection', [0.34274988807662354, 0.32490737289807736, 0.7154095230770581, 0.5826210826210826, 0.8481979635330333, 0.1518020364669666, 0.41737891737891736, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2458430733787651]), ('RandomUnderSampler', [0.29595982211493427, 0.26138356609288727, 0.704824057376745, 0.6276353276353277, 0.7820127871181625, 0.21798721288183756, 0.37236467236467236, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03549340977256321]), ('TomekLinks', [0.32285748341759096, 0.30039772934596198, 0.709385545037925, 0.59002849002849, 0.8287426000473597, 0.1712573999526403, 0.40997150997151, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2953349810914361]), ('ADASYN', [0.30746849586882086, 0.27630319673353188, 0.7089299676915115, 0.6202279202279203, 0.7976320151551028, 0.202367984844897, 0.3797720797720797, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0227008461765124]), ('RandomOverSampler', [0.29082645981131466, 0.25870876314141655, 0.6996062938473547, 0.6122507122507123, 0.7869618754439971, 0.21303812455600285, 0.38774928774928774, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7935464551869309]), ('SMOTE', [0.32217478364045271, 0.29809634949439912, 0.710764258675696, 0.5977207977207977, 0.8238077196305943, 0.17619228036940565, 0.4022792022792022, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8953182579968981]), ('SMOTEENN', [0.33993548506036636, 0.32440349739484331, 0.7109194729985637, 0.5678062678062678, 0.8540326781908595, 0.14596732180914043, 0.43219373219373225, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7658827137537578]), ('SMOTETomek', [0.32217478364045271, 0.29809634949439912, 0.710764258675696, 0.5977207977207977, 0.8238077196305943, 0.17619228036940565, 0.4022792022792022, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8246876343281038])]), ('SVM (polynomial)', [('None', [0.32388774444662471, 0.30728826131150244, 0.7033464820582983, 0.5603988603988603, 0.8462941037177363, 0.15370589628226378, 0.43960113960113956, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2902196460741209]), ('ClusterCentroids', [0.34478628231371822, 0.3348972389454018, 0.7053611109424512, 0.5381766381766381, 0.8725455837082643, 0.12745441629173573, 0.4618233618233618, 0.0, 0.0, 0.0, 0.0, 0.0, 0.017474728204437184]), ('EditedNearestNeighbours', [0.31910744066443547, 0.30040639043584827, 0.7031595866618362, 0.5678062678062678, 0.8385129055174045, 0.1614870944825953, 0.43219373219373225, 0.0, 0.0, 0.0, 0.0, 0.0, 0.18236444030158339]), ('InstanceHardnessThreshold', [0.216118660758787, 0.19460289706670797, 0.6476382690637961, 0.5102564102564102, 0.7850201278711817, 0.21497987212881836, 0.48974358974358967, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01672882989051061]), ('NearMiss', [0.2388290935841664, 0.17833433168044854, 0.6827091838008248, 0.7165242165242165, 0.6488941510774331, 0.3511058489225669, 0.28347578347578345, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01708230901864738]), ('NeighbourhoodCleaningRule', [0.32543880131612896, 0.30770486362375071, 0.7055915070975455, 0.5678062678062678, 0.8433767463888231, 0.15662325361117688, 0.43219373219373225, 0.0, 0.0, 0.0, 0.0, 0.0, 0.17592846379359628]), ('OneSidedSelection', [0.33993548506036636, 0.32440349739484331, 0.7109407848621678, 0.5678062678062678, 0.8540753019180677, 0.14592469808193229, 0.43219373219373214, 0.0, 0.0, 0.0, 0.0, 0.0, 0.22908194437065532]), ('RandomUnderSampler', [0.24412003183903239, 0.2029913286387709, 0.6774332056240652, 0.6273504273504272, 0.7275159838977031, 0.27248401610229694, 0.3726495726495726, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01714962400204456]), ('TomekLinks', [0.32388774444662471, 0.30728826131150244, 0.7033464820582983, 0.5603988603988603, 0.8462941037177363, 0.15370589628226378, 0.43960113960113956, 0.0, 0.0, 0.0, 0.0, 0.0, 0.29314407459103364]), ('ADASYN', [0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8451948572560113]), ('RandomOverSampler', [0.33519669791404516, 0.31471588149587182, 0.714280169712327, 0.59002849002849, 0.8385318493961638, 0.16146815060383612, 0.40997150997151, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6407806847654385]), ('SMOTE', [0.32543880131612896, 0.30770486362375071, 0.7055867711278556, 0.5678062678062678, 0.8433672744494436, 0.15663272555055646, 0.43219373219373214, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8137168013368878]), ('SMOTEENN', [0.33583058469545024, 0.32100998977424677, 0.7077177820819782, 0.5603988603988603, 0.8550367037650959, 0.1449632962349041, 0.43960113960113956, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4877199962327623]), ('SMOTETomek', [0.32543880131612896, 0.30770486362375071, 0.7055867711278556, 0.5678062678062678, 0.8433672744494436, 0.15663272555055646, 0.43219373219373214, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6649461600877089])])])]
    visualise_dataset_balancer_results(results, colors=("#64B3DE", "#1f78b4", "#B9B914", "#FBAC44", "#bc1659", "#33a02c", "#ff7f00", "grey", "#b15928", "#e31a1c"))