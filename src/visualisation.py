import os
from collections import OrderedDict
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


def plot_percentage_difference_graph(results, datasets, name_suffix="", parameter="Balanced Accuracy", x_label="Feature selection approach", difference_from="no feature selection", figsize=(16, 5), legend_y=-0.31, label_rotation=0, y_label_pos=-0.4):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = "raw_dump_{0}.txt".format(current_time)
    with open(os.path.dirname(os.path.realpath(__file__)) + "/../results/" + file_name, "wb") as output_file:
        output_file.write(str(results))
    patterns = (None, "////")

    colors = ["#64B3DE", "#1f78b4", "#FBAC44", "#B9B914", "#bc1659", "#33a02c", "#6ABF20", "#ff7f00", "#6a3d9a", "#5a2add", "#b15928", "#e31a1c", "grey"]
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
            for result_tuple in data_balancer_results:
                value = result_tuple[0][2] - no_feature_selection[x][0][2]
                classifier_arr[index][x].append(value)
                mean_classification += value
                x += 1
            mean_classification /= float(len(data_balancer_results))
            classifier_arr[index][x].append(mean_classification)
        index += 1

    fig = plt.figure(figsize=figsize)

    classifiers = np.arange(len(classifier_arr[0]))

    bar_width = 0.2
    opacity = 0.9
    num_columns = 1 if len(results) == 1 else 2
    subplt_val = (100 * round(len(results) / 2.0)) + (10 * num_columns) + 1
    print(subplt_val)
    plt.subplots_adjust(hspace=0.4, wspace=0.1)
    ax1 = plt.subplot(subplt_val)

    for i in range(len(classifier_arr[0])):
        if i + 1 != len(classifier_arr[0]):
            label = results[0][0][1][i][1]
        else:
            label = "Mean classification"
        data_balancers = np.arange(len(classifier_arr[0][i])) * 3
        plt.bar(data_balancers + (i * bar_width), classifier_arr[0][i], bar_width,
                alpha=opacity,
                color=colors[i],
                hatch=patterns[i % len(patterns)],
                label=label)

        feature_selection_labels = [results[0][i][0] for i in range(1, len(results[0]))]
        plt.xticks(data_balancers + (bar_width / 2) * len(classifiers), feature_selection_labels, rotation=label_rotation)
        plt.title(datasets[0].replace("_", " "))
        plt.ylabel("Difference in {0} from {1}".format(parameter, difference_from), y=y_label_pos)

    vertical_plt = 0
    for z in range(1, len(results)):
        ax2 = plt.subplot(subplt_val + z, sharey=ax1)
        color = iter(cm.Set1(np.linspace(0, 1, len(no_feature_selection) + 1)))
        for i in range(len(classifier_arr[z])):
            if i + 1 != len(classifier_arr[z]):
                label = results[z][0][1][i][1]
            else:
                label = "Mean classification"
            print(classifier_arr[0])
            print(classifier_arr[z])
            data_balancers = np.arange(len(classifier_arr[z][i])) * 3
            plt.bar(data_balancers + (i * bar_width), classifier_arr[z][i], bar_width,
                    alpha=opacity,
                    color=colors[i],
                    hatch=patterns[i % len(patterns)],
                    label=label)

        feature_selection_labels = [results[z][i][0] for i in range(1, len(results[z]))]
        plt.xticks(data_balancers + (bar_width / 2) * len(classifiers), feature_selection_labels, rotation=label_rotation)
        plt.title(datasets[z].replace("_", " "))

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    legend = plt.legend(loc='lower center', bbox_to_anchor=(-0.08, legend_y), fancybox=True, frameon=True, ncol=7)
    legend.get_frame().set_facecolor('#ffffff')

    plt.xlabel(x_label, x=0, y=-2)
    feature_selection_labels = [results[0][i][0] for i in range(1, len(results[0]))]


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
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = "raw_dump_{0}.txt".format(current_time)
    with open(os.path.dirname(os.path.realpath(__file__)) + "/../results/" + file_name, "wb") as output_file:
        output_file.write(str(dataset_results))
    sns.set(style='ticks')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    markers = ["s", "o", "^", "*"]
    colors = ["#64B3DE", "#1f78b4", "#B9B914", "#FBAC44", "#bc1659", "#33a02c", "#6ABF20", "#ff7f00", "#6a3d9a", "grey", "#b15928", "#e31a1c", "black"]
    color_dict = {}
    index = 0
    for (_, classifier_description) in dataset_results[0][1]:
        color_dict[classifier_description] = colors[index]
        index += 1

    hatches = [None, "////", ".."]

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
                classifier_labels.append(mpatches.Patch(facecolor=color_dict[classifier_description], hatch=hatches[i % len(hatches)], label=classifier_description, alpha=0.8, edgecolor="black"))
            ax.scatter(result_arr[3] - median_true_pos, result_arr[4] - median_true_neg, marker=markers[data_set_index], hatch=hatches[i % len(hatches)], s=200, alpha=0.8, color=colors[i], edgecolor="black", zorder=data_set_index, lw=0.8)
            i += 1
        data_set_index += 1

    plt.legend(handles=data_set_labels + classifier_labels)
    sns.despine()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/classifier_dataset_plt_{0}.png".format(current_time), bbox_inches='tight')
    plt.close(fig)


def visualise_dataset_balancer_results(results, range=(-0.3, 0.3), colors=("#64B3DE", "#1f78b4", "#B9B914", "#FBAC44", "#bc1659", "#33a02c", "#6ABF20", "#ff7f00", "#6a3d9a", "grey", "#b15928", "#e31a1c"), exclude=("SVM (linear)", "SVM (polynomial)", "SVM (RDF)", "Logistic regression")):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = "raw_dump_{0}.txt".format(current_time)
    with open(os.path.dirname(os.path.realpath(__file__)) + "/../results/" + file_name, "wb") as output_file:
        output_file.write(str(results))
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
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = "raw_dump_{0}.txt".format(current_time)
    with open(os.path.dirname(os.path.realpath(__file__)) + "/../results/" + file_name, "wb") as output_file:
        output_file.write(str(dataset_results))
    sns.set(style='ticks')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    markers = ["s", "o", "^", "d", "*"]
    colors = ["#64B3DE", "#1f78b4", "#B9B914", "#FBAC44", "#bc1659", "#33a02c", "#6ABF20", "#ff7f00", "#6a3d9a", "grey", "#b15928", "#e31a1c"]
    hatches = [None, "////", ".."]
    color_dict = {}
    index = 0
    for (classifier_description, result_arr) in dataset_results[0][1]:
        print(classifier_description)
        color_dict[classifier_description] = colors[index]
        index += 1

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
    results = [('India_CHW_attrition', [('AdaBoost', [('None', [0.56219038990037973, 0.39490837708129511, 0.7800277306195201, 0.7060325307455217, 0.8540229304935189, 0.1459770695064813, 0.2939674692544783, 0.46368396960541974, 0.3599051151992329, 0.02836887307475543, 0.07424883273825872, 0.07424883273825872, 22.32626880097851, 0.016011645699223413]), ('ClusterCentroids', [0.45232157709809134, 0.28350437723454353, 0.7333083521692015, 0.9049577344441393, 0.561658969894264, 0.438341030105736, 0.0950422655558607, 0.0, 0.058644526879820996, 0.0, 0.0014135310812047972, 0.0014135310812047972, 5.6133670289692645, 0.011374308807575215]), ('EditedNearestNeighbours', [0.52901011419768551, 0.28452199700469771, 0.7777588521430712, 0.8966822301565505, 0.6588354741295919, 0.3411645258704083, 0.1033177698434496, 0.0, 0.052187481599246306, 0.0, 0.0006060606060606061, 0.0006060606060606061, 6.663850211387474, 0.011218402789129894]), ('InstanceHardnessThreshold', [0.5153540361032104, 0.29453251824274801, 0.7709723743809401, 0.8757020354603435, 0.6662427133015368, 0.3337572866984632, 0.12429796453965637, 0.02945588818700601, 0.04774387833211363, 0.00021180844710256476, 0.0008074704751441912, 0.0008074704751441912, 6.266419704863513, 0.013103788644161809]), ('NearMiss', [0.40928895945564631, 0.34957099595364899, 0.7153647118530716, 0.7891525527175074, 0.6415768709886357, 0.35842312901136425, 0.2108474472824926, 0.5195800909396077, 0.20282231811643578, 0.09410284233813644, 0.053064176508285266, 0.053064176508285266, 15.334195646996099, 0.015772284456742114]), ('NeighbourhoodCleaningRule', [0.54360906588234814, 0.28478632609584681, 0.7859219825809483, 0.8853806951692148, 0.6864632699926818, 0.31353673000731824, 0.1146193048307852, 0.0, 0.05451552392728864, 0.0, 0.0008074704751441912, 0.0008074704751441912, 5.7102496292403275, 0.015788195281385822]), ('OneSidedSelection', [0.56725220092954087, 0.39422040781901485, 0.7859521371126134, 0.7316414904330313, 0.8402627837921957, 0.15973721620780443, 0.2683585095669688, 0.48225640086667276, 0.343073998368116, 0.032814831638361054, 0.06698465012664408, 0.06698465012664408, 28.309332739272577, 0.014393529961263404]), ('RandomUnderSampler', [0.52561237623245149, 0.28459556914233858, 0.7765270323649567, 0.8736842747718881, 0.6793697899580252, 0.32063021004197473, 0.12631572522811194, 0.0, 0.05017942312059959, 0.0, 0.0010094906771643932, 0.0010094906771643932, 7.857388010660543, 0.011472204075837154]), ('TomekLinks', [0.57116797955519971, 0.39443143858718499, 0.7877221230517836, 0.7328505599804694, 0.8425936861230978, 0.15740631387690213, 0.26714944001953067, 0.4858921541700998, 0.3482613706143118, 0.03260335966218319, 0.0647673105679148, 0.0647673105679148, 30.77632750773546, 0.014952361081514022]), ('ADASYN', [0.57404270157945325, 0.39358799403613887, 0.7901597645688635, 0.7415362080014648, 0.8387833211362623, 0.16121667886373767, 0.25846379199853514, 0.46086545210412283, 0.38107367872073755, 0.03419150242679654, 0.075456071286887, 0.075456071286887, 57.85163294738444, 0.01431909555435545]), ('RandomOverSampler', [0.55697070936306858, -1.0, 0.7883497726895629, 0.7893564039183373, 0.7873431414607884, 0.21265685853921146, 0.21064359608166253, 0.0, 0.0, 0.0, 0.0, 0.0, 14.481295893425036, 0.01436575075940242]), ('SMOTE', [0.56303012337887581, -1.0, 0.7902717209026077, 0.78208733864323, 0.7984561031619855, 0.20154389683801446, 0.21791266135677012, 0.0, 0.0, 0.0, 0.0, 0.0, 12.01394694245361, 0.013809557578087757]), ('SMOTEENN', [0.55630038711080232, -1.0, 0.7682261989139544, 0.6519411639049101, 0.8845112339229985, 0.11548876607700137, 0.34805883609508986, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5071827105809157, 0.015604279120462205]), ('SMOTETomek', [0.54646847187690961, 0.28519489935970732, 0.7873199488322999, 0.8567408221184656, 0.7178990755461344, 0.28210092445386564, 0.14325917788153436, 0.0, 0.054622858152269915, 0.0, 0.0008074704751441912, 0.0008074704751441912, 6.252688994854907, 0.011425025051251965])]), ('Artificial neural network', [('None', [0.58076085003399525, 0.64901569667531278, 0.78461744973563, 0.6912887179956666, 0.8779461814755932, 0.12205381852440678, 0.3087112820043334, 0.4945460648783912, 0.795484223719518, 0.044143639437757086, 0.19551771491348532, 0.19551771491348532, 83.91292483412035, 0.017487108911462785]), ('ClusterCentroids', [0.46443646093706198, 0.57503211041631141, 0.743314652936121, 0.8662223442888095, 0.6204069615834321, 0.37959303841656783, 0.13377765571119046, 0.7806805212243279, 0.4849165131518072, 0.25002733826263235, 0.08656168940156854, 0.08656168940156854, 145.97270792349528, 0.015311461604851575]), ('EditedNearestNeighbours', [0.50240518844086479, 0.64793840317249385, 0.7643081182498278, 0.8583594250663736, 0.6702568114332821, 0.32974318856671797, 0.14164057493362628, 0.8218377124721536, 0.5923548758842877, 0.2799855317502376, 0.10229302084286979, 0.10229302084286979, 16.19410610120652, 0.012974430033026393]), ('InstanceHardnessThreshold', [0.47970106680745922, 0.63658507959747357, 0.7517943728731004, 0.8658225762153254, 0.6377661695308754, 0.36223383046912455, 0.13417742378467454, 0.8569416216546125, 0.600181021357492, 0.34847284259048966, 0.1049113491409564, 0.1049113491409564, 37.131722386517744, 0.012641258930358695]), ('NearMiss', [0.36882357465367388, 0.47929224677910748, 0.6881050833352237, 0.6358588910250541, 0.7403512756453932, 0.2596487243546067, 0.36414110897494584, 0.4526595257713082, 0.47748201983496114, 0.1718952566011389, 0.11719918215386493, 0.11719918215386493, 11.320954755296649, 0.07916751944348545]), ('NeighbourhoodCleaningRule', [0.52464790080578227, 0.65967343004184154, 0.7759828004629853, 0.8593585400836156, 0.6926070608423549, 0.307392939157645, 0.14064145991638438, 0.8117415850346364, 0.614175520057873, 0.24896577249518423, 0.1025023650401294, 0.1025023650401294, 7.898234812738944, 0.016854481050708164]), ('OneSidedSelection', [0.58446950109154694, 0.64641539815132842, 0.7902746486432287, 0.7155128322499925, 0.8650364650364651, 0.13496353496353497, 0.28448716775000765, 0.5304861301840154, 0.7690243184360831, 0.051019759255053386, 0.17068204705666942, 0.17068204705666942, 20.134736962637362, 0.01730248264642301]), ('RandomUnderSampler', [0.5600373405783724, 0.60343246519106242, 0.7897735983856474, 0.7893472489242882, 0.7901999478470066, 0.20980005215299333, 0.21065275107571182, 0.6600329579785772, 0.6369249922191099, 0.11982486688369039, 0.10471726326711221, 0.10471726326711221, 16.90317374694617, 0.0145574193713234]), ('TomekLinks', [0.5755245315259736, 0.67123318099634355, 0.7849302931193815, 0.704405383136501, 0.865455203102262, 0.1345447968977381, 0.295594616863499, 0.5532826756995941, 0.7476340205751969, 0.059599599599599595, 0.16667276999603284, 0.16667276999603284, 27.607865528170805, 0.015662466592606488]), ('ADASYN', [0.5670082425605375, 0.64206351812396301, 0.79574339786531, 0.8242491379047271, 0.767237657825893, 0.23276234217410688, 0.175750862095273, 0.6935042265555861, 0.6386226562697151, 0.11950639715345597, 0.11621898745765813, 0.11621898745765813, 25.15767782653499, 0.02042511008553332]), ('RandomOverSampler', [0.58766171952832635, 0.62967454542125945, 0.8013520764595937, 0.7832921358601116, 0.8194120170590761, 0.18058798294092412, 0.2167078641398883, 0.6424590314016296, 0.6776763317939789, 0.09082460611872377, 0.11782904574445359, 0.11782904574445359, 19.188272252446648, 0.016208572608694988]), ('SMOTE', [0.58224031632805029, 0.63137792111545976, 0.797130446046387, 0.7677548902926546, 0.8265060018001193, 0.17349399819988054, 0.23224510970734533, 0.6311465104214349, 0.681170582347053, 0.08626592979534158, 0.12085873844182, 0.12085873844182, 20.3790912984911, 0.015596058032506495]), ('SMOTEENN', [0.5584991814322593, 0.68207889685858192, 0.7667625769367373, 0.6404424913790473, 0.8930826624944271, 0.1069173375055728, 0.35955750862095276, 0.49274619304830786, 0.8248091789268259, 0.04710794828441887, 0.263309225182337, 0.263309225182337, 28.203077383142347, 0.01478167806859575]), ('SMOTETomek', [0.57710858213987615, 0.63894596868023013, 0.7945568841374776, 0.764727028594098, 0.8243867396808572, 0.17561326031914268, 0.23527297140590195, 0.6019042387622449, 0.6789451636510461, 0.07748605748605748, 0.12267447892825535, 0.12267447892825535, 21.726338067806864, 0.018020229165259572])]), ('Bernoulli Naive Bayes', [('None', [0.57411594646799968, 0.69881328291582268, 0.7824578201848505, 0.6931056791479753, 0.8718099612217259, 0.12819003877827406, 0.3068943208520248, 0.5819353657420123, 0.7413992143403908, 0.06975883446471681, 0.13921816350819374, 0.13921816350819374, 0.2030004088191845, 0.01282285489035382]), ('ClusterCentroids', [0.42643407141895606, 0.59545415461177365, 0.7125756317226016, 0.9390613079434832, 0.4860899555017202, 0.5139100444982798, 0.060938692056516836, 0.9162611004302846, 0.4391941521353286, 0.45802155096272745, 0.05125209801946962, 0.05125209801946962, 0.11434872829869391, 0.00981839824023422]), ('EditedNearestNeighbours', [0.56142487447184453, 0.67447225833902535, 0.7954365284114706, 0.8904403552137691, 0.7004327016091721, 0.2995672983908278, 0.10955964478623088, 0.7976233635448137, 0.6379747815041933, 0.20007536948713422, 0.08796789648753395, 0.08796789648753395, 0.12520472402979976, 0.010320506203043457]), ('InstanceHardnessThreshold', [0.54497866998931033, 0.67247148107679255, 0.7863553385572223, 0.8484659281638134, 0.7242447489506314, 0.2757552510493687, 0.15153407183618664, 0.8087259299948122, 0.626440053498877, 0.23002481473069705, 0.08050779700326528, 0.08050779700326528, 0.1207555075129676, 0.010885496745116813]), ('NearMiss', [0.36641735266050812, 0.63695980944710096, 0.6907271877593542, 0.676574201226769, 0.7048801742919389, 0.29511982570806095, 0.3234257987732308, 0.629361896914767, 0.6659259259259259, 0.2578598766834061, 0.2840873996765236, 0.2840873996765236, 0.11734599777294837, 0.011740185778273071]), ('NeighbourhoodCleaningRule', [0.57084647067044492, 0.67747704671328679, 0.8003649373029149, 0.8860050657633739, 0.7147248088424559, 0.28527519115754413, 0.11399493423662609, 0.7677658762855137, 0.6494055399937753, 0.18122172592760824, 0.09200036619976197, 0.09200036619976197, 0.20020820296758757, 0.012030305510186448]), ('OneSidedSelection', [0.57550022823468527, 0.69206268307460339, 0.7858554980609359, 0.7098519942628704, 0.8618590018590019, 0.13814099814099814, 0.2901480057371296, 0.5865751167261741, 0.7280616751204988, 0.07759204582733993, 0.12912813940004272, 0.12912813940004272, 0.25506322509924456, 0.012822606068215244]), ('RandomUnderSampler', [0.56942245123546664, 0.67936684219970134, 0.795051245868733, 0.8086941926821081, 0.7814082990553578, 0.21859170094464211, 0.19130580731789193, 0.6668631938722572, 0.6937601467013232, 0.11538092714563303, 0.11037016692605817, 0.11037016692605817, 0.17904660307264444, 0.013068448069945764]), ('TomekLinks', [0.57563694385718767, 0.69212473770639737, 0.7859084922315773, 0.7098519942628704, 0.8619649902002843, 0.1380350097997157, 0.2901480057371296, 0.5865751167261741, 0.7281676634617812, 0.07748622572151984, 0.12912813940004272, 0.12912813940004272, 0.30904023684342974, 0.012826353660235853]), ('ADASYN', [0.55102696824634467, 0.63858643130021231, 0.7898821560026464, 0.892462388232781, 0.6873019237725121, 0.312698076227488, 0.10753761176721903, 0.8063224388904148, 0.5879094220270691, 0.21457154633625222, 0.07182947297750923, 0.07182947297750923, 0.3017560509620125, 0.008807208311572237]), ('RandomOverSampler', [0.57704126687846591, 0.68295711316144792, 0.801707717078252, 0.8400024413317464, 0.7634129928247576, 0.23658700717524248, 0.15999755866825352, 0.6660685403887822, 0.7079497985380337, 0.10574927869045518, 0.11560743385516799, 0.11560743385516799, 0.22480406512977424, 0.013560098670657985]), ('SMOTE', [0.56812700955756779, 0.68649267243684031, 0.7937744838420153, 0.7925978821447099, 0.7949510855393208, 0.20504891446067916, 0.20740211785529006, 0.6539674692544784, 0.7124997266173737, 0.10490423196305548, 0.11903933595776496, 0.11903933595776496, 0.17942443128792243, 0.015982789363365713]), ('SMOTEENN', [0.56262347624826348, 0.74601786767629186, 0.7567399653697858, 0.587792730934725, 0.9256871998048469, 0.07431280019515314, 0.4122072690652751, 0.5318923372699808, 0.8875785028726205, 0.047106602400720045, 0.33170679605724923, 0.33170679605724923, 0.18065702666429692, 0.014367864393974255]), ('SMOTETomek', [0.56778217811183451, 0.688230018243571, 0.7897006358762179, 0.757885806707559, 0.8215154650448767, 0.17848453495512318, 0.24211419329244102, 0.6406475632457506, 0.7216024427789135, 0.09791522615052027, 0.12388232780982027, 0.12388232780982027, 0.3640717465094288, 0.015267622827799796])]), ('Extreme Learning Machine', [('None', [0.46805365660992782, 0.52550703332756865, 0.7158247718937251, 0.5421624095944337, 0.8894871341930165, 0.11051286580698344, 0.45783759040556626, 0.29843266501876775, 0.6551247045364691, 0.020747301923772513, 0.1864487778082944, 0.1864487778082944, 1.5481947307503565, 0.016538993735268893]), ('ClusterCentroids', [0.39354296303860031, 0.43921113428519076, 0.7047716483028372, 0.8430162653727608, 0.5665270312329136, 0.4334729687670864, 0.15698373462723916, 0.5746760657938906, 0.2868621562739209, 0.19805418864242394, 0.04842930818761634, 0.04842930818761634, 1.1388777787604643, 0.01916291090952015]), ('EditedNearestNeighbours', [0.43081221074164783, 0.50079012423182334, 0.7264836085788637, 0.8188031371112942, 0.6341640800464331, 0.36583591995356696, 0.18119686288870582, 0.6801953065397175, 0.41113600995953936, 0.23489371724665842, 0.06254814001037566, 0.06254814001037566, 1.2389033463100532, 0.0144768323567339]), ('InstanceHardnessThreshold', [0.42825313716345365, 0.5205749128062388, 0.7242142825295482, 0.8442167902590864, 0.6042117748000101, 0.39578822519998985, 0.15578320974091367, 0.7148817479935304, 0.49031165619400907, 0.25913577442989205, 0.09906313894229302, 0.09906313894229302, 1.2184716441183052, 0.010434372637049332]), ('NearMiss', [0.36415252398243408, 0.48477548969221845, 0.6915798058911619, 0.7596820165400225, 0.6234775952423011, 0.3765224047576989, 0.24031798345997743, 0.49535841801702835, 0.4750775986070104, 0.21996399761105642, 0.12006591595715461, 0.12006591595715461, 1.779393903170785, 0.01412887712972058]), ('NeighbourhoodCleaningRule', [0.4425827917437461, 0.50364298983854938, 0.7328165009167321, 0.7917739326802771, 0.6738590691531868, 0.3261409308468131, 0.2082260673197229, 0.6289310018615154, 0.43749430102371284, 0.1828139063433181, 0.06496872043699838, 0.06496872043699838, 1.8016337473147057, 0.01414092041595658]), ('OneSidedSelection', [0.47302517438469033, 0.52370904169198751, 0.7232859072898169, 0.5720095211938112, 0.8745622933858228, 0.1254377066141772, 0.4279904788061888, 0.3196161005828679, 0.6452805746923392, 0.026462092344445284, 0.179791266135677, 0.179791266135677, 2.315644835037953, 0.014958540960121463]), ('RandomUnderSampler', [0.46636561850195929, 0.49063404150284406, 0.7432076216597275, 0.752428819921267, 0.733986423398188, 0.2660135766018119, 0.24757118007873294, 0.4769751899661265, 0.4608837408837409, 0.08976674153144742, 0.07203454484421253, 0.07203454484421253, 1.051348816542486, 0.016897832170658874]), ('TomekLinks', [0.47203739826823038, 0.52379942678194313, 0.7228241468280565, 0.5714034605877506, 0.8742448330683626, 0.1257551669316375, 0.42859653941224934, 0.3196148799169947, 0.6450689344806991, 0.02635627223862518, 0.179791266135677, 0.179791266135677, 1.5189713533767037, 0.015278002714601639]), ('ADASYN', [0.46790017603722522, 0.45506939222881915, 0.7457381180857278, 0.7911532240837378, 0.700323012087718, 0.2996769879122821, 0.20884677591626233, 0.45318929476029174, 0.38922249981073503, 0.07780351780351781, 0.05185693795965699, 0.05185693795965699, 2.8520889573640793, 0.01670717170549834]), ('RandomOverSampler', [0.48151267899894651, 0.49293260985973886, 0.7498106924568626, 0.7441484329701852, 0.7554729519435401, 0.24452704805645986, 0.25585156702981476, 0.4596283072416003, 0.4731637519872815, 0.066689210218622, 0.07525954408129636, 0.07525954408129636, 5.24594071125648, 0.018340138572154937]), ('SMOTE', [0.48406030219048318, 0.49727054284031119, 0.7504542399530848, 0.7374939729622509, 0.7634145069439188, 0.23658549305608131, 0.2625060270377491, 0.46225823186548265, 0.48248080012785893, 0.0694410376763318, 0.07747932497177211, 0.07747932497177211, 2.5812269787655473, 0.019653526772001887]), ('SMOTEENN', [0.46986536579577276, 0.56213286288690612, 0.7225910348551188, 0.5746412768165033, 0.870540792893734, 0.12945920710626593, 0.4253587231834966, 0.34504196038939244, 0.6807398995634291, 0.030170506641094875, 0.22214959260276482, 0.22214959260276482, 1.7783574162346365, 0.014502239789761194]), ('SMOTETomek', [0.48596636073417265, 0.50121280198283669, 0.7506044106648337, 0.7292196893405354, 0.771989131989132, 0.228010868010868, 0.27078031065946473, 0.45438920931368054, 0.4906304623951683, 0.06637124519477461, 0.08373584790503219, 0.08373584790503219, 1.8144135980326912, 0.018232044728662772])]), ('Gaussian Naive Bayes', [('None', [0.51801241466315251, 0.78736349996087152, 0.7081945205136959, 0.4503677255943117, 0.9660213154330801, 0.03397868456691986, 0.5496322744056883, 0.4469407061552076, 0.96591549532726, 0.033343763931999226, 0.5496322744056883, 0.5496322744056883, 0.22956641118234608, 0.016784102658197738]), ('ClusterCentroids', [0.52824427384319805, 0.79395403907858986, 0.7417418142618072, 0.5705996521102261, 0.9128839764133881, 0.08711602358661184, 0.4294003478897739, 0.5689834904940646, 0.9126723362017479, 0.0864811029516912, 0.4291983276877536, 0.4291983276877536, 0.14217368848586412, 0.01589647866736909]), ('EditedNearestNeighbours', [0.47549461117090247, 0.77397809480343105, 0.7133439493560338, 0.5185712105953798, 0.9081166881166882, 0.09188331188331189, 0.48142878940462025, 0.5185712105953798, 0.908010868010868, 0.09177749177749178, 0.48142878940462025, 0.48142878940462025, 0.16756575695073175, 0.016031979367592725]), ('InstanceHardnessThreshold', [0.47398622680060476, 0.76885733369208564, 0.7283619289146213, 0.5984625713326619, 0.8582612864965807, 0.1417387135034194, 0.40153742866733805, 0.5984625713326619, 0.8582612864965807, 0.1417387135034194, 0.40153742866733805, 0.40153742866733805, 0.18638854485243428, 0.01329585005279647]), ('NearMiss', [0.20105585037970028, 0.63934922067827016, 0.6005057409358103, 0.47559278586468917, 0.7254186960069314, 0.27458130399306874, 0.5244072141353108, 0.47559278586468917, 0.7252070557952912, 0.27458130399306874, 0.5242051939332907, 0.5242051939332907, 0.21295706271621384, 0.016755377796174028]), ('NeighbourhoodCleaningRule', [0.47377851094008222, 0.77331045567365653, 0.7038249317243452, 0.4794207940431505, 0.92822906940554, 0.07177093059446, 0.5205792059568494, 0.479219384174067, 0.92822906940554, 0.0716651104886399, 0.5203777960877659, 0.5203777960877659, 0.16700641020027737, 0.014031307567402151]), ('OneSidedSelection', [0.50677726622203123, 0.78365451889435134, 0.7004663770027197, 0.4332179804083127, 0.9677147735971267, 0.032285226402873464, 0.5667820195916873, 0.4332179804083127, 0.9677147735971267, 0.032073586191233246, 0.5667820195916873, 0.5667820195916873, 0.27204396209905707, 0.01646144807508734]), ('RandomUnderSampler', [0.48858383545864553, 0.77691927785086234, 0.6922206367843474, 0.4201129115932741, 0.9643283619754208, 0.0356716380245792, 0.5798870884067259, 0.4197094815221704, 0.9643283619754208, 0.035565817918759096, 0.5796856785376422, 0.5796856785376422, 0.214924083695981, 0.015077495258113]), ('TomekLinks', [0.50677726622203123, 0.78368672381016102, 0.7004663770027197, 0.4332179804083127, 0.9677147735971267, 0.032285226402873464, 0.5667820195916873, 0.4332179804083127, 0.9677147735971267, 0.032073586191233246, 0.5665799993896671, 0.5665799993896671, 0.42396370675730094, 0.01646144807508734]), ('ADASYN', [0.48464679665700977, 0.77480127772023344, 0.6884784116879373, 0.41040556623638197, 0.9665512571394924, 0.033448742860507565, 0.589594433763618, 0.4087936769507767, 0.9665512571394924, 0.03313128254304725, 0.5893930238945345, 0.5893930238945345, 0.3519458253705731, 0.015039442560143397]), ('RandomOverSampler', [0.51967262014464277, 0.78900507507759288, 0.71183231747831, 0.461664988251091, 0.9619996467055292, 0.03800035329447094, 0.5383350117489091, 0.4604553083707162, 0.9613647260706085, 0.03789453318865083, 0.5379315816778052, 0.5379315816778052, 0.28600569569774353, 0.017687685121537546]), ('SMOTE', [0.52508553586046414, 0.79170375729910225, 0.7152006485470143, 0.46871860599957277, 0.9616826910944557, 0.038317308905544196, 0.5312813940004273, 0.46831578626140563, 0.9616826910944557, 0.038317308905544196, 0.5300723244529891, 0.5300723244529891, 0.3290064707737959, 0.016460720573146643]), ('SMOTEENN', [0.53854054610099833, 0.76820043005671346, 0.782336354943428, 0.8270835240623761, 0.7375891858244801, 0.2624108141755201, 0.17291647593762396, 0.8270835240623761, 0.7374831974831976, 0.2624108141755201, 0.17291647593762396, 0.17291647593762396, 0.3038241950727449, 0.013153030958936733]), ('SMOTETomek', [0.52422038221576528, 0.79142627657158338, 0.7312231582162273, 0.5330446458543135, 0.9294016705781413, 0.07059832942185884, 0.4669553541456865, 0.532237785712106, 0.9287667499432205, 0.07038668921021864, 0.4655424333974183, 0.4655424333974183, 0.3697719718679764, 0.01956309178112765])])])]
    visualise_dataset_balancer_results(results, colors=("#64B3DE", "#1f78b4", "#B9B914", "#bc1659", "#33a02c", "#6ABF20", "#ff7f00", "#6a3d9a", "grey", "#b15928", "#e31a1c"))