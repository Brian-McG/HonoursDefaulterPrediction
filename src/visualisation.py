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
    results = [[('Default without balancer', [([0.27351080507102726, 0.23992056857802946, 0.5877849446986712, 0.19924026590693256, 0.9763296234904096, 0.023670376509590335, 0.8007597340930674, 0.0, 0.00032362459546925567, 0.0, 0.0, 0.0, 0.8915189990468249, 0.0336660656748246], 'AdaBoost', 'Default without balancer'), ([0.25067024904765756, 0.23091407355425217, 0.5890215724993529, 0.2117758784425451, 0.9662672665561608, 0.033732733443839286, 0.7882241215574549, 0.09990503323836657, 0.9279990528060621, 0.013949009393006551, 0.6444444444444445, 0.6444444444444445, 4.0771229317653495, 0.035625403856293376], 'Artificial neural network', 'Default without balancer'), ([0.36058558464036428, 0.35337838792421605, 0.7084543423062644, 0.5320037986704654, 0.8849048859420634, 0.1150951140579367, 0.4679962013295347, 0.44529914529914527, 0.841771252663983, 0.07878601310284948, 0.43057929724596394, 0.43057929724596394, 0.019972697727906146, 0.0338291738775351], 'Bernoulli Naive Bayes', 'Default without balancer'), ([0.22089905294400478, 0.21985937985677004, 0.6026633622821167, 0.27891737891737894, 0.9264093456468544, 0.07359065435314548, 0.7210826210826212, 0.26400759734093066, 0.8897639908437919, 0.07034809377219986, 0.6244064577397911, 0.6244064577397911, 0.015059315076611623, 0.04173431892883681], 'Decision Tree', 'Default without balancer'), ([0.14592408900324846, 0.090047294099441788, 0.5283779483716337, 0.06486229819563152, 0.9918935985476359, 0.008106401452364038, 0.9351377018043685, 0.0, 0.930289683479359, 0.0, 0.6170940170940171, 0.6170940170940171, 0.0496203249487567, 0.023407251962758016], 'Extreme Learning Machine', 'Default without balancer'), ([0.11933838688211777, 0.044807024336493195, 0.578238835446192, 0.9132003798670466, 0.24327729102533743, 0.7567227089746625, 0.08679962013295346, 0.8883190883190881, 0.18291893598547637, 0.6853974267898019, 0.06951566951566951, 0.06951566951566951, 0.0065192802801504746, 0.03179280530144338], 'Gaussian Naive Bayes', 'Default without balancer'), ([0.14939909959856049, 0.097245018736305491, 0.5308291117763058, 0.07236467236467235, 0.9892935511879392, 0.010706448812060934, 0.9276353276353276, 0.007502374169040836, 0.9507048701554978, 0.0016228589470360722, 0.711301044634378, 0.711301044634378, 0.011739856256063145, 0.02315601300883657], 'K-nearest neighbours', 'Default without balancer'), ([0.28921766453768433, 0.24597330016558996, 0.5876994566228128, 0.19420702754036087, 0.9811918857052647, 0.018808114294735178, 0.8057929724596392, 0.03760683760683761, 0.939689004657037, 0.0032441392375088794, 0.6269705603038936, 0.6269705603038936, 0.017343996252100622, 0.03801131392998844], 'Logistic regression', 'Default without balancer'), ([0.19950549163271428, 0.18728152744889984, 0.5743978853198208, 0.19192782526115862, 0.956867945378483, 0.043132054621517084, 0.8080721747388414, 0.07236467236467237, 0.8871686794537849, 0.014264740705659482, 0.5574548907882241, 0.5574548907882241, 0.28561321995857064, 0.029351172981929536], 'Random forest', 'Default without balancer'), ([0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.14245660277991334, 0.0], 'SVM (RDF)', 'Default without balancer'), ([0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.14924131902152538, 0.0], 'SVM (linear)', 'Default without balancer'), ([0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1413658803302393, 0.0], 'SVM (polynomial)', 'Default without balancer')], 'Default without balancer'), ('Default with balancer', [([0.29415443986089923, 0.25657403505422305, 0.7056442616621005, 0.6392212725546058, 0.772067250769595, 0.22793274923040496, 0.36077872744539413, 0.0, 0.00032362459546925567, 0.0, 0.0, 0.0, 1.2419904598008908, 0.03474851204787913], 'AdaBoost', 'Default with balancer'), ([0.23035990980746279, 0.22502261351292918, 0.6344533361938051, 0.40541310541310543, 0.8634935669745047, 0.1365064330254953, 0.5945868945868945, 0.26125356125356125, 0.7759349593495934, 0.07360959823190465, 0.47977207977207975, 0.47977207977207975, 7.989110585270197, 0.04648864128515435], 'Artificial neural network', 'Default with balancer'), ([0.35480622586638888, 0.34691100502948991, 0.706668881733212, 0.5320037986704654, 0.8813339647959587, 0.11866603520404136, 0.4679962013295347, 0.45764482431149095, 0.8356113347541242, 0.08235061962270109, 0.4281101614434948, 0.4281101614434948, 0.01969297370483473, 0.03487917152720766], 'Bernoulli Naive Bayes', 'Default with balancer'), ([0.19280390875516371, 0.14884992524835994, 0.6455806993718431, 0.6267806267806267, 0.6643807719630593, 0.3356192280369406, 0.3732193732193732, 0.6267806267806267, 0.5748472649775042, 0.32556002841581816, 0.30826210826210826, 0.30826210826210826, 0.005489634262852251, 0.03263570361033896], 'Decision Tree', 'Default with balancer'), ([0.25477799588867861, 0.21333581080722344, 0.6837580684080015, 0.6339981006647674, 0.7335180361512353, 0.26648196384876466, 0.3660018993352327, 0.4056030389363723, 0.3868813639592707, 0.08849790828005366, 0.14710351377018047, 0.14710351377018047, 0.03125191492727123, 0.05581952535197998], 'Extreme Learning Machine', 'Default with balancer'), ([0.24565466212826151, 0.19186709701625773, 0.6808359751093983, 0.700664767331434, 0.6610071828873628, 0.3389928171126371, 0.29933523266856593, 0.6757834757834758, 0.6184955402952087, 0.30880101034020047, 0.27939221272554604, 0.27939221272554604, 0.0088615161826886, 0.07480143879755118], 'Gaussian Naive Bayes', 'Default with balancer'), ([0.22933384833151907, 0.21852610210033704, 0.642138440984443, 0.44738841405508073, 0.8368884679138052, 0.1631115320861946, 0.5526115859449193, 0.3036087369420703, 0.76850580156287, 0.1031273186518273, 0.40569800569800557, 0.40569800569800557, 0.02193348263351484, 0.054501377821656434], 'K-nearest neighbours', 'Default with balancer'), ([0.30919966908213153, 0.27279872661153287, 0.7138820559145763, 0.644349477682811, 0.7834146341463413, 0.2165853658536586, 0.355650522317189, 0.4400759734093067, 0.5003741416054938, 0.0833278080353619, 0.15451092117758783, 0.15451092117758783, 0.034039294383071315, 0.03392835486756706], 'Logistic regression', 'Default with balancer'), ([0.2325531177976512, 0.18595014741800234, 0.6727439614699856, 0.6492877492877492, 0.696200173652222, 0.303799826347778, 0.35071225071225065, 0.4029439696106363, 0.41601547083432, 0.13034177914594683, 0.1768281101614435, 0.1768281101614435, 0.26517283976331313, 0.03929204361559816], 'Random forest', 'Default with balancer'), ([0.30495990213706986, 0.28735929975109231, 0.6933661995237493, 0.5494776828110162, 0.8372547162364826, 0.16274528376351724, 0.45052231718898383, 0.0, 0.0, 0.0, 0.0, 0.0, 0.016712604797146254, 0.03796618985199927], 'SVM (RDF)', 'Default with balancer'), ([0.31563290711648812, 0.29430886607798368, 0.7032434128755859, 0.5770180436847103, 0.8294687820664614, 0.17053121793353854, 0.4229819563152896, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7848783406770953, 0.039686984436063936], 'SVM (linear)', 'Default with balancer'), ([0.31563290711648812, 0.29430886607798368, 0.7032434128755859, 0.5770180436847103, 0.8294687820664614, 0.17053121793353854, 0.4229819563152896, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7561983774597983, 0.039686984436063936], 'SVM (polynomial)', 'Default with balancer')], 'Default with balancer'), ('Tuned', [([0.28109883282144249, 0.23383150252665832, 0.7038851255245603, 0.6739791073124407, 0.73379114373668, 0.26620885626331986, 0.32602089268755935, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8066098786651773, 0.03627451185664151], 'AdaBoost', 'Tuned'), ([0.30886046511779158, 0.27006366136657539, 0.7154159546408342, 0.6542260208926877, 0.776605888388981, 0.22339411161101905, 0.34577397910731245, 0.43266856600189935, 0.49939537453626964, 0.08754913568553162, 0.15194681861348527, 0.15194681861348527, 3.7049437799992693, 0.03146413388928601], 'Artificial neural network', 'Tuned'), ([0.27403561004518751, 0.23743490930012509, 0.692699410522443, 0.6217473884140551, 0.7636514326308311, 0.23634856736916884, 0.37825261158594486, 0.5320037986704654, 0.658913884284474, 0.1514089509827137, 0.3036087369420703, 0.3036087369420703, 0.005049622337350772, 0.027169872726643684], 'Bernoulli Naive Bayes', 'Tuned'), ([0.22405167361565922, 0.19497931189350645, 0.6571166152703763, 0.5551756885090219, 0.7590575420317309, 0.240942457968269, 0.4448243114909782, 0.3778727445394112, 0.30609203567763826, 0.1124808587891704, 0.1668566001899335, 0.1668566001899335, 0.005515191781506988, 0.03367426685809685], 'Decision Tree', 'Tuned'), ([0.28646337240337888, 0.25419079628509345, 0.6968010098904408, 0.6092117758784426, 0.7843902439024389, 0.21560975609756097, 0.39078822412155745, 0.3980056980056979, 0.4543247296550635, 0.07553713789565081, 0.15213675213675212, 0.15213675213675212, 0.13989642362774493, 0.041660103291566275], 'Extreme Learning Machine', 'Tuned'), ([0.24565466212826151, 0.19186709701625773, 0.6808359751093983, 0.700664767331434, 0.6610071828873628, 0.3389928171126371, 0.29933523266856593, 0.6757834757834758, 0.6184955402952087, 0.30880101034020047, 0.27939221272554604, 0.27939221272554604, 0.009094854315603381, 0.07480143879755118], 'Gaussian Naive Bayes', 'Tuned'), ([0.21828801292955666, 0.18733868284578389, 0.6546333637595775, 0.5618233618233618, 0.747443365695793, 0.2525566343042071, 0.43817663817663816, 0.2911680911680911, 0.12094403662483227, 0.05932433499092273, 0.044729344729344735, 0.044729344729344735, 0.0019018214964292568, 0.034128140666507004], 'K-nearest neighbours', 'Tuned'), ([0.30747701436807384, 0.27079886345813314, 0.7130698371127765, 0.644349477682811, 0.7817901965427421, 0.2182098034572579, 0.355650522317189, 0.4451092117758784, 0.5068561054542584, 0.0862499013339648, 0.156980056980057, 0.156980056980057, 0.04728493121656705, 0.033424914669010077], 'Logistic regression', 'Tuned'), ([0.28398899892570934, 0.24727539730028891, 0.6986079554845834, 0.6293447293447293, 0.7678711816244376, 0.2321288183755624, 0.37065527065527065, 0.3734093067426401, 0.2924729655063541, 0.07326071513142317, 0.09924026590693258, 0.09924026590693258, 0.4555988926551012, 0.04466896102408589], 'Random forest', 'Tuned'), ([0.27797563845183088, 0.24558202546788602, 0.6920067603381203, 0.6044634377967711, 0.7795500828794695, 0.22044991712053041, 0.3955365622032289, 0.37587844254510916, 0.28688452127239716, 0.07262135922330097, 0.0846153846153846, 0.0846153846153846, 0.08391157219289651, 0.054525825898009396], 'SVM (RDF)', 'Tuned'), ([0.29534786626698711, 0.26496400028273742, 0.700556451701767, 0.60664767331434, 0.7944652300891941, 0.20553476991080588, 0.39335232668566, 0.4255460588793922, 0.49260557265766836, 0.08625779461678112, 0.14188034188034188, 0.14188034188034188, 6.627032388721406, 0.03946330758275901], 'SVM (linear)', 'Tuned'), ([0.11201108924211427, 0.053578432182109927, 0.5160640786143983, 0.03732193732193732, 0.9948062199068591, 0.005193780093140737, 0.9626780626780626, 0.0, 1.0, 0.0, 1.0, 1.0, 1.3578049593056887, 0.010763438771560668], 'SVM (polynomial)', 'Tuned')], 'Tuned')], [('Default without balancer', [([0.49965307065104514, 0.48550318787198493, 0.7276170067533139, 0.5516707864139888, 0.9035632270926389, 0.09643677290736113, 0.44832921358601113, 0.0, 0.056847435670965084, 0.0, 0.0010088803442277763, 0.0010088803442277763, 6.814052550081924, 0.029394296331655394], 'AdaBoost', 'Default without balancer'), ([0.5780892494518709, 0.57551688571969617, 0.7797600237799278, 0.6708730812658305, 0.8886469662940252, 0.11135303370597488, 0.3291269187341695, 0.499561780951509, 0.7852252252252252, 0.03842346548228901, 0.19370624675760628, 0.19370624675760628, 54.261768759394776, 0.013249725003887715], 'Artificial neural network', 'Default without balancer'), ([0.5658876706659397, 0.56536328642867795, 0.7790076461215613, 0.6912820043333637, 0.8667332879097586, 0.13326671209024152, 0.30871799566663616, 0.57607250755287, 0.7303959421606482, 0.07451787922376157, 0.13257468949311849, 0.13257468949311849, 0.36131358401234337, 0.015237027916987422], 'Bernoulli Naive Bayes', 'Default without balancer'), ([0.48239388024526825, 0.48214275979502985, 0.7434606954564303, 0.6777710641154749, 0.8091503267973855, 0.19084967320261437, 0.32222893588452495, 0.6430663126735635, 0.7652195052195051, 0.16671377259612555, 0.26573102627483286, 0.26573102627483286, 2.107717696478707, 0.011187192763515203], 'Decision Tree', 'Default without balancer'), ([0.10424671488742937, 0.064599693181331519, 0.5260026944565775, 0.09180872165766425, 0.9601966672554908, 0.039803332744509215, 0.9081912783423357, 0.02703774909212976, 0.2431500407970996, 0.006034942505530741, 0.10957673410845616, 0.10957673410845616, 1.0097343960028853, 0.014268154514428187], 'Extreme Learning Machine', 'Default without balancer'), ([0.5137323656899212, 0.46784164369733289, 0.707223458471015, 0.45075162501144383, 0.963695291930586, 0.03630470806941395, 0.5492483749885563, 0.45075162501144383, 0.963695291930586, 0.03609306785777374, 0.5486423143824957, 0.5486423143824957, 0.4819827039995583, 0.014854199788649275], 'Gaussian Naive Bayes', 'Default without balancer'), ([0.4731336459268487, 0.47182785463524562, 0.7414899593982587, 0.6951081815130152, 0.7878717372835021, 0.21212826271649798, 0.3048918184869846, 0.5304702615276633, 0.6354413236766178, 0.09579730991495698, 0.16283316549177576, 0.16283316549177576, 0.7557001064661897, 0.012072740003388273], 'K-nearest neighbours', 'Default without balancer'), ([0.5455843536417605, 0.53656481214350993, 0.7539443146764988, 0.5998730507491837, 0.9080155786038139, 0.09198442139618612, 0.4001269492508163, 0.41705514358082335, 0.7286976051681934, 0.023182678476796125, 0.16847935548841894, 0.16847935548841894, 5.958693716985895, 0.013834213797706186], 'Logistic regression', 'Default without balancer'), ([0.56897764208495316, 0.56843918365042445, 0.780503798511618, 0.6931099514785316, 0.8678976455447044, 0.13210235445529564, 0.30689004852146845, 0.5381390948762549, 0.6791675709322768, 0.05641776230011523, 0.1604077024016601, 0.1604077024016601, 2.26486303031473, 0.011401350500857279], 'Random forest', 'Default without balancer'), ([0.3926531557515382, 0.35727439213162038, 0.6581717648307305, 0.38578473557325516, 0.930558794088206, 0.06944120591179415, 0.6142152644267448, 0.0, 0.0, 0.0, 0.0, 0.0, 38.68809332559363, 0.013026999005316729], 'SVM (RDF)', 'Default without balancer'), ([0.3926531557515382, 0.35727439213162038, 0.6581717648307305, 0.38578473557325516, 0.930558794088206, 0.06944120591179415, 0.6142152644267448, 0.0, 0.0, 0.0, 0.0, 0.0, 38.60693673316754, 0.013026999005316729], 'SVM (linear)', 'Default without balancer'), ([0.3926531557515382, 0.35727439213162038, 0.6581717648307305, 0.38578473557325516, 0.930558794088206, 0.06944120591179415, 0.6142152644267448, 0.0, 0.0, 0.0, 0.0, 0.0, 40.636050972961655, 0.013026999005316729], 'SVM (polynomial)', 'Default without balancer')], 'Default without balancer'), ('Default with balancer', [([0.54126778275315857, 0.5205629611138759, 0.7845654358218779, 0.8524996185419146, 0.7166312531018413, 0.28336874689815866, 0.1475003814580854, 0.0, 0.05208182972888855, 0.0, 0.0008068601422075742, 0.0008068601422075742, 21.40371099195673, 0.01205654532400059], 'AdaBoost', 'Default with balancer'), ([0.58557168346109878, 0.58121116334979206, 0.8015113619663113, 0.7941841374469774, 0.8088385864856452, 0.19116141351435467, 0.20581586255302267, 0.653349812322622, 0.6715453268394445, 0.09886474710004121, 0.1244896090817541, 0.1244896090817541, 125.5658280681323, 0.01094092154374934], 'Artificial neural network', 'Default with balancer'), ([0.54853375710649688, 0.51487506839935115, 0.7884518730404638, 0.8968860813573803, 0.6800176647235471, 0.31998233527645287, 0.10311391864261954, 0.8038731728157709, 0.5703478268184151, 0.21666893784540844, 0.06376209222130673, 0.06376209222130673, 0.3246508204013689, 0.013206110772875092], 'Bernoulli Naive Bayes', 'Default with balancer'), ([0.5101244278071162, 0.50743546697264152, 0.7620956435884462, 0.7356831151393086, 0.7885081720375838, 0.2114918279624162, 0.2643168848606915, 0.7050151057401813, 0.7045665833901129, 0.1810050386520975, 0.22417223595471328, 0.22417223595471328, 7.2142063177984985, 0.007499124985206122], 'Decision Tree', 'Default with balancer'), ([0.2142015895542585, 0.20812623636150976, 0.6111796812472128, 0.5962342457810735, 0.626125116713352, 0.37387488328664803, 0.4037657542189264, 0.06921053434648601, 0.090293486764075, 0.02456069514893044, 0.024011108059446427, 0.024011108059446427, 1.3395148317626078, 0.017170497181719303], 'Extreme Learning Machine', 'Default with balancer'), ([0.53050542695792491, 0.51349072688974851, 0.7785477979346828, 0.832519759528823, 0.7245758363405422, 0.2754241636594578, 0.16748024047117704, 0.832519759528823, 0.724364196128902, 0.2754241636594578, 0.16748024047117704, 0.16748024047117704, 0.42662069130672836, 0.009924256801906122], 'Gaussian Naive Bayes', 'Default with balancer'), ([0.47909123065604819, 0.47597610082800829, 0.7467303011642802, 0.7209356403918338, 0.7725249619367265, 0.22747503806327338, 0.2790643596081663, 0.5532710793737984, 0.6251737451737451, 0.10828812005282595, 0.15415911379657604, 0.15415911379657604, 0.9656587445402588, 0.010458945258294968], 'K-nearest neighbours', 'Default with balancer'), ([0.54567952799576935, 0.53917385685940944, 0.7831751693578601, 0.7885367267844611, 0.777813611931259, 0.22218638806874103, 0.21146327321553904, 0.5607330098568769, 0.5737305372599489, 0.06838098602804484, 0.08030394580243523, 0.08030394580243523, 9.838882585036352, 0.009913988551555388], 'Logistic regression', 'Default with balancer'), ([0.57383027840557876, 0.57300597850910351, 0.7913599513208541, 0.7508138789709787, 0.8319060236707295, 0.16809397632927045, 0.24918612102902135, 0.6138167170191339, 0.6134246851893911, 0.08944356120826709, 0.11259422014709025, 0.11259422014709025, 3.2905774157867236, 0.009738560710839382], 'Random forest', 'Default with balancer'), ([0.38658537199724091, 0.38429211443550998, 0.6858556617451234, 0.5405499099758919, 0.8311614135143547, 0.16883858648564531, 0.4594500900241081, 0.0, 0.0, 0.0, 0.0, 0.0, 92.55380275250451, 0.013994923571773727], 'SVM (RDF)', 'Default with balancer'), ([0.3832799326254992, 0.38168181888822916, 0.6855775360200452, 0.547615734383106, 0.8235393376569847, 0.17646066234301527, 0.452384265616894, 0.0, 0.0, 0.0, 0.0, 0.0, 101.18850350364399, 0.013286473227342095], 'SVM (linear)', 'Default with balancer'), ([0.33422836230414826, 0.33408458383771045, 0.668563459242329, 0.5784759986572675, 0.7586509198273902, 0.2413490801726096, 0.4215240013427324, 0.0, 0.0, 0.0, 0.0, 0.0, 9.372083462001356, 0.013482388794182863], 'SVM (polynomial)', 'Default with balancer')], 'Default with balancer'), ('Tuned', [([0.57470805780025425, 0.57394345370948019, 0.7916436656191409, 0.7502139216942841, 0.8330734095439977, 0.16692659045600222, 0.24978607830571578, 0.4618712807836675, 0.378432045490869, 0.03598993951935128, 0.07546278494918979, 0.07546278494918979, 170.52692358815082, 0.011075666950320385], 'AdaBoost', 'Tuned'), ([0.58557168346109878, 0.58121116334979206, 0.8015113619663113, 0.7941841374469774, 0.8088385864856452, 0.19116141351435467, 0.20581586255302267, 0.653349812322622, 0.6715453268394445, 0.09886474710004121, 0.1244896090817541, 0.1244896090817541, 153.16033861498573, 0.01094092154374934], 'Artificial neural network', 'Tuned'), ([0.57680942257103651, 0.56200495152635066, 0.8021614024457459, 0.8494711465104214, 0.75485165838107, 0.24514834161892982, 0.15052885348957856, 0.6779663706551924, 0.6899639134933252, 0.12066974537562773, 0.10553327840336903, 0.10553327840336903, 0.28311522406047135, 0.010787312941008789], 'Bernoulli Naive Bayes', 'Tuned'), ([0.26207227182466036, 0.20999847625009535, 0.5922357190635136, 0.3038450975006866, 0.8806263406263407, 0.11937365937365936, 0.6961549024993134, 0.2160950898715249, 0.11241914183090655, 0.02984984984984985, 0.010088803442277763, 0.010088803442277763, 0.1386055676953772, 0.011547133553874442], 'Decision Tree', 'Tuned'), ([0.47409950640373322, 0.47108753733178066, 0.7439038906890337, 0.7154951325948306, 0.7723126487832369, 0.22768735121676298, 0.2845048674051695, 0.44976593731880743, 0.48576862576862573, 0.06192326780562074, 0.07969666453050138, 0.07969666453050138, 3.183190055267629, 0.011602113244297267], 'Extreme Learning Machine', 'Tuned'), ([0.53050542695792491, 0.51349072688974851, 0.7785477979346828, 0.832519759528823, 0.7245758363405422, 0.2754241636594578, 0.16748024047117704, 0.832519759528823, 0.724364196128902, 0.2754241636594578, 0.16748024047117704, 0.16748024047117704, 0.38717920157303826, 0.009924256801906122], 'Gaussian Naive Bayes', 'Tuned'), ([0.28867296084348965, 0.19631482830407884, 0.6272281714897673, 0.9400750709512039, 0.3143812720283308, 0.6856187279716691, 0.05992492904879612, 0.8585480179437882, 0.11262960439431029, 0.4872490978373331, 0.008879123561903018, 0.008879123561903018, 0.0212887086982884, 0.019240182695753217], 'K-nearest neighbours', 'Tuned'), ([0.53781145042363665, 0.5235841215121616, 0.7818633198409278, 0.8248637431719003, 0.7388628965099553, 0.26113710349004465, 0.17513625682809972, 0.5688003906130794, 0.5440941782118253, 0.08192865134041605, 0.070818151301535, 0.070818151301535, 12.911011939889107, 0.012087252777336407], 'Logistic regression', 'Tuned'), ([0.446675340525089, 0.42314725133415737, 0.7310407252010238, 0.7798651164210076, 0.6822163339810398, 0.31778366601896013, 0.22013488357899233, 0.07000457749702461, 0.05652947064711771, 0.00031746031746031746, 0.0008068601422075742, 0.0008068601422075742, 1.0181351725087258, 0.02413584107838282], 'Random forest', 'Tuned'), ([0.47847257757956307, 0.4631110826148615, 0.7512253698125387, 0.7966144832005858, 0.7058362564244917, 0.2941637435755083, 0.20338551679941408, 0.6075467667612683, 0.549380641145347, 0.149573102514279, 0.11823186548262077, 0.11823186548262077, 119.0916579152289, 0.012530913694898702], 'SVM (RDF)', 'Tuned'), ([0.52313522135990709, 0.51520318862333891, 0.7723299680257206, 0.7841002166681926, 0.7605597193832487, 0.23944028061675118, 0.2158997833318075, 0.6969104946748451, 0.566650179591356, 0.14734583322818615, 0.10491318013976625, 0.10491318013976625, 300.509312042914, 0.013561624613643025], 'SVM (linear)', 'Tuned'), ([0.071476671514822177, 0.022789084541794884, 0.5087921575179439, 0.025418535811285056, 0.9921657792246027, 0.007834220775397248, 0.974581464188715, 0.011098904452378773, 0.0027526686350215755, 0.0017996147407912113, 0.0004034300711037871, 0.0004034300711037871, 251.4207233441872, 0.004052788182729256], 'SVM (polynomial)', 'Tuned')], 'Tuned')], [('Default without balancer', [([0.35008331083046368, 0.34749480033776708, 0.6657142857142857, 0.48333333333333334, 0.8480952380952381, 0.1519047619047619, 0.5166666666666667, 0.0, 0.0004761904761904762, 0.0, 0.0, 0.0, 1.1626381879437537, 0.03715913359286836], 'AdaBoost', 'Default without balancer'), ([0.38732751116391118, 0.38558769648986785, 0.6858730158730159, 0.5222222222222223, 0.8495238095238097, 0.15047619047619049, 0.47777777777777786, 0.41333333333333333, 0.768095238095238, 0.09333333333333334, 0.37999999999999995, 0.37999999999999995, 5.501137158649908, 0.03523183842636496], 'Artificial neural network', 'Default without balancer'), ([0.37903625130963697, 0.37893081761006292, 0.6912698412698411, 0.5777777777777778, 0.8047619047619047, 0.1952380952380952, 0.4222222222222222, 0.41, 0.7042857142857143, 0.12, 0.26666666666666666, 0.26666666666666666, 0.03914808136990677, 0.040133155794224884], 'Bernoulli Naive Bayes', 'Default without balancer'), ([0.24954984509353892, 0.24942527436507481, 0.6256349206349207, 0.4822222222222223, 0.769047619047619, 0.23095238095238094, 0.5177777777777778, 0.4822222222222223, 0.769047619047619, 0.23095238095238094, 0.5177777777777778, 0.5177777777777778, 0.04022461637812335, 0.025447122813018], 'Decision Tree', 'Default without balancer'), ([0.25263901615200951, 0.22525697883399495, 0.5954761904761904, 0.26999999999999996, 0.920952380952381, 0.07904761904761905, 0.73, 0.03222222222222222, 0.5957142857142858, 0.007142857142857143, 0.2833333333333333, 0.2833333333333333, 0.09311559937361924, 0.040023613589314846], 'Extreme Learning Machine', 'Default without balancer'), ([0.31295870286746003, 0.29827527873122683, 0.6696825396825395, 0.6822222222222223, 0.6571428571428571, 0.3428571428571429, 0.31777777777777777, 0.6322222222222222, 0.6204761904761904, 0.30666666666666664, 0.2677777777777777, 0.2677777777777777, 0.008771461146483675, 0.0490304074182532], 'Gaussian Naive Bayes', 'Default without balancer'), ([0.28914890077822464, 0.27096890817064356, 0.6188888888888888, 0.33777777777777773, 0.9, 0.09999999999999999, 0.6622222222222223, 0.1211111111111111, 0.7233333333333333, 0.02666666666666667, 0.40111111111111103, 0.40111111111111103, 0.01202783113161369, 0.026827979588260255], 'K-nearest neighbours', 'Default without balancer'), ([0.36624580398041279, 0.36121894104735786, 0.6694444444444444, 0.47222222222222215, 0.8666666666666667, 0.13333333333333333, 0.5277777777777778, 0.22666666666666666, 0.7242857142857143, 0.034761904761904765, 0.2822222222222222, 0.2822222222222222, 0.03998906460186043, 0.03707886904430117], 'Logistic regression', 'Default without balancer'), ([0.29173267612495768, 0.26957613218400089, 0.6170634920634921, 0.32555555555555554, 0.9085714285714287, 0.09142857142857141, 0.6744444444444445, 0.10999999999999999, 0.5647619047619047, 0.02, 0.21333333333333335, 0.21333333333333335, 0.2957999035455221, 0.021143012059111174], 'Random forest', 'Default without balancer'), ([0.28951156470877193, 0.24329569520641811, 0.6000000000000001, 0.25333333333333335, 0.9466666666666668, 0.05333333333333334, 0.7466666666666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24827579822721402, 0.03310579938827654], 'SVM (RDF)', 'Default without balancer'), ([0.28951156470877193, 0.24329569520641811, 0.6000000000000001, 0.25333333333333335, 0.9466666666666668, 0.05333333333333334, 0.7466666666666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23909954220849375, 0.03310579938827654], 'SVM (linear)', 'Default without balancer'), ([0.28951156470877193, 0.24329569520641811, 0.6000000000000001, 0.25333333333333335, 0.9466666666666668, 0.05333333333333334, 0.7466666666666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.22689129914401407, 0.03310579938827654], 'SVM (polynomial)', 'Default without balancer')], 'Default without balancer'), ('Default with balancer', [([0.3948692379761381, 0.38297912719473493, 0.7121428571428572, 0.71, 0.7142857142857143, 0.2857142857142857, 0.29, 0.0011111111111111111, 0.0004761904761904762, 0.0, 0.0, 0.0, 1.1575689446767263, 0.026769218906736794], 'AdaBoost', 'Default with balancer'), ([0.37153091699943969, 0.35979227995003882, 0.6996825396825397, 0.6955555555555556, 0.7038095238095238, 0.2961904761904762, 0.30444444444444446, 0.6, 0.6166666666666666, 0.21523809523809523, 0.2133333333333333, 0.2133333333333333, 7.490807896307313, 0.032219094382554395], 'Artificial neural network', 'Default with balancer'), ([0.39296596335645489, 0.37676463043510716, 0.7125396825396825, 0.7322222222222222, 0.692857142857143, 0.3071428571428571, 0.2677777777777778, 0.5766666666666667, 0.5657142857142857, 0.19952380952380952, 0.15555555555555556, 0.15555555555555556, 0.06200052785331837, 0.03996140112967655], 'Bernoulli Naive Bayes', 'Default with balancer'), ([0.30991982462547485, 0.2652830055512902, 0.6672222222222222, 0.8077777777777778, 0.5266666666666666, 0.47333333333333333, 0.1922222222222222, 0.8077777777777778, 0.5266666666666666, 0.47333333333333333, 0.1922222222222222, 0.1922222222222222, 0.017820432672690425, 0.0252458308999361], 'Decision Tree', 'Default with balancer'), ([0.29961761819268234, 0.25948697658651815, 0.662142857142857, 0.79, 0.5342857142857143, 0.46571428571428575, 0.20999999999999996, 0.5211111111111112, 0.280952380952381, 0.23666666666666666, 0.06888888888888889, 0.06888888888888889, 0.04357587116676276, 0.031689036137722275], 'Extreme Learning Machine', 'Default with balancer'), ([0.34186799253722389, 0.31601623844664756, 0.6864285714285714, 0.75, 0.6228571428571429, 0.37714285714285717, 0.25000000000000006, 0.7222222222222223, 0.5885714285714285, 0.34761904761904755, 0.2188888888888889, 0.2188888888888889, 0.008650817584014092, 0.05023339329534916], 'Gaussian Naive Bayes', 'Default with balancer'), ([0.34811911614544627, 0.33528983069387541, 0.6877777777777778, 0.6888888888888888, 0.6866666666666666, 0.31333333333333335, 0.31111111111111117, 0.528888888888889, 0.4995238095238095, 0.18476190476190477, 0.15333333333333332, 0.15333333333333332, 0.00634510679672277, 0.023094283512700254], 'K-nearest neighbours', 'Default with balancer'), ([0.39737151045835001, 0.38617029352484766, 0.7131746031746031, 0.7077777777777778, 0.7185714285714285, 0.28142857142857136, 0.2922222222222222, 0.5366666666666667, 0.5633333333333332, 0.16666666666666666, 0.16, 0.16, 0.02635765010489224, 0.03017793874502471], 'Logistic regression', 'Default with balancer'), ([0.34112839047458171, 0.30539311232965211, 0.6857936507936507, 0.7877777777777778, 0.5838095238095239, 0.41619047619047617, 0.2122222222222222, 0.5777777777777778, 0.29428571428571426, 0.23666666666666666, 0.05333333333333332, 0.05333333333333332, 0.29706661064136713, 0.02929141339135223], 'Random forest', 'Default with balancer'), ([0.4029601266630829, 0.38532089528926522, 0.7181746031746031, 0.7444444444444445, 0.6919047619047619, 0.30809523809523814, 0.25555555555555554, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11716462070680207, 0.02103009876747651], 'SVM (RDF)', 'Default with balancer'), ([0.39045344974865709, 0.38040500049915654, 0.709047619047619, 0.6966666666666667, 0.7214285714285714, 0.2785714285714286, 0.30333333333333334, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5802302329950186, 0.03905729651661385], 'SVM (linear)', 'Default with balancer'), ([0.37708268947860552, 0.36048217200227911, 0.7042063492063493, 0.7255555555555556, 0.6828571428571427, 0.3171428571428571, 0.27444444444444444, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10342825939110478, 0.024136119824266028], 'SVM (polynomial)', 'Default with balancer')], 'Default with balancer'), ('Tuned', [([0.39589655331040757, 0.38547220344922534, 0.7120634920634922, 0.7022222222222222, 0.7219047619047619, 0.2780952380952381, 0.29777777777777775, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24493923398079373, 0.03044103461713074], 'AdaBoost', 'Tuned'), ([0.39739836624705177, 0.3769521019627784, 0.715793650793651, 0.7544444444444444, 0.677142857142857, 0.32285714285714284, 0.24555555555555553, 0.5855555555555555, 0.5114285714285715, 0.19571428571428573, 0.1311111111111111, 0.1311111111111111, 5.33465306724938, 0.02375178183500381], 'Artificial neural network', 'Tuned'), ([0.39302601157311895, 0.3766024953764962, 0.7126190476190476, 0.7333333333333334, 0.6919047619047619, 0.30809523809523803, 0.26666666666666666, 0.5877777777777777, 0.5633333333333334, 0.2019047619047619, 0.15444444444444444, 0.15444444444444444, 0.016739570997931885, 0.04090528853810334], 'Bernoulli Naive Bayes', 'Tuned'), ([0.28177866971472709, 0.26849399942609287, 0.6520634920634921, 0.6522222222222223, 0.6519047619047619, 0.34809523809523807, 0.3477777777777778, 0.41, 0.43476190476190474, 0.16285714285714284, 0.1877777777777778, 0.1877777777777778, 0.010016937390216649, 0.03493181639168329], 'Decision Tree', 'Tuned'), ([0.39540904700240054, 0.38422036266670717, 0.7121428571428572, 0.7066666666666667, 0.7176190476190477, 0.28238095238095234, 0.29333333333333333, 0.4366666666666667, 0.4761904761904762, 0.11714285714285715, 0.12, 0.12, 0.05313267260104206, 0.028773097790515796], 'Extreme Learning Machine', 'Tuned'), ([0.34186799253722389, 0.31601623844664756, 0.6864285714285714, 0.75, 0.6228571428571429, 0.37714285714285717, 0.25000000000000006, 0.7222222222222223, 0.5885714285714285, 0.34761904761904755, 0.2188888888888889, 0.2188888888888889, 0.006872557634467628, 0.05023339329534916], 'Gaussian Naive Bayes', 'Tuned'), ([0.37637269131737372, 0.36324558903398724, 0.7028571428571428, 0.7066666666666667, 0.699047619047619, 0.3009523809523809, 0.29333333333333333, 0.3522222222222222, 0.38428571428571434, 0.0938095238095238, 0.07555555555555556, 0.07555555555555556, 0.00569248448965561, 0.0253352427971159], 'K-nearest neighbours', 'Tuned'), ([0.39873882606631544, 0.37872945559433435, 0.7164285714285715, 0.7533333333333333, 0.6795238095238094, 0.32047619047619047, 0.24666666666666667, 0.6044444444444445, 0.5247619047619048, 0.19999999999999998, 0.13999999999999999, 0.13999999999999999, 0.020354953686053345, 0.022780128190219046], 'Logistic regression', 'Tuned'), ([0.39620952382272795, 0.38717032417937264, 0.7114285714285714, 0.6933333333333334, 0.7295238095238096, 0.2704761904761905, 0.3066666666666667, 0.3766666666666667, 0.47428571428571425, 0.07904761904761905, 0.11666666666666664, 0.11666666666666664, 2.3288670060181906, 0.04545852727698058], 'Random forest', 'Tuned'), ([0.38631451079339146, 0.37198884121505316, 0.7084920634920634, 0.718888888888889, 0.6980952380952381, 0.3019047619047619, 0.2811111111111111, 0.21666666666666665, 0.6776190476190477, 0.032857142857142856, 0.24333333333333332, 0.24333333333333332, 1.3973131602002622, 0.030896979315360385], 'SVM (RDF)', 'Tuned'), ([0.37345130275372523, 0.36024404730689114, 0.7013492063492063, 0.7055555555555556, 0.6971428571428572, 0.3028571428571429, 0.29444444444444445, 0.1677777777777778, 0.660952380952381, 0.02476190476190476, 0.2577777777777778, 0.2577777777777778, 1.3362786864280618, 0.025290700890333466], 'SVM (linear)', 'Tuned'), ([0.38515705552061291, 0.36546507013562124, 0.7091269841269842, 0.7444444444444445, 0.6738095238095237, 0.3261904761904762, 0.25555555555555554, 0.5333333333333333, 0.5480952380952381, 0.1580952380952381, 0.15444444444444444, 0.15444444444444444, 0.722188866641773, 0.030653830827797816], 'SVM (polynomial)', 'Tuned')], 'Tuned')], [('Default without balancer', [([0.6946222453861125, 0.69414607470707512, 0.8484882944344662, 0.8501322051824433, 0.8468443836864888, 0.15315561631351107, 0.14986779481755685, 0.018420588753745814, 0.0, 0.0, 0.0, 0.0, 1.1258590068167136, 0.021149706506105787], 'AdaBoost', 'Default without balancer'), ([0.72000359839668882, 0.71986537023400088, 0.8606162870037455, 0.8544861625242376, 0.8667464114832536, 0.1332535885167464, 0.14551383747576238, 0.7742816851753922, 0.8093984962406015, 0.0792207792207792, 0.09226158998766083, 0.09226158998766083, 4.050758238249805, 0.023786461455081537], 'Artificial neural network', 'Default without balancer'), ([0.70403170574030283, 0.70244923784343383, 0.8486360154267695, 0.7990833774017275, 0.8981886534518112, 0.10181134654818864, 0.2009166225982725, 0.7470826723074211, 0.8511847801321485, 0.0722032353611301, 0.14010223867442273, 0.14010223867442273, 0.020311586400461785, 0.015861413758456892], 'Bernoulli Naive Bayes', 'Default without balancer'), ([0.62137494516905145, 0.62072134868778961, 0.8114800034314911, 0.8057465185968624, 0.8172134882661198, 0.18278651173388016, 0.19425348140313767, 0.8057465185968624, 0.8172134882661198, 0.18278651173388016, 0.19425348140313767, 0.19425348140313767, 0.020310077098179186, 0.025385858908352776], 'Decision Tree', 'Default without balancer'), ([0.68878151282448519, 0.6878059769850724, 0.8439605274905451, 0.8263352723426758, 0.861585782638414, 0.1384142173615858, 0.17366472765732413, 0.525559668605676, 0.6588630667578036, 0.05048986101617681, 0.07163758152652917, 0.07163758152652917, 0.1263989345533324, 0.028661687869426793], 'Extreme Learning Machine', 'Default without balancer'), ([0.50271089449079986, 0.46377969205331748, 0.7226311355524526, 0.5114049004054292, 0.933857370699476, 0.06614262930052403, 0.48859509959457076, 0.49841353781068215, 0.9295055821371611, 0.06352244246981088, 0.47340031729243776, 0.47340031729243776, 0.005964360140856033, 0.048329681854638096], 'Gaussian Naive Bayes', 'Default without balancer'), ([0.60828165984128824, 0.60674071923024098, 0.801117779907198, 0.7449673893883307, 0.8572681704260652, 0.14273182957393482, 0.2550326106116693, 0.5897761325577296, 0.7337092731829573, 0.05309865573023467, 0.1476291203948528, 0.1476291203948528, 0.00712783096055943, 0.025185752158315193], 'K-nearest neighbours', 'Default without balancer'), ([0.71083325519143115, 0.71003864303093123, 0.8568472916583078, 0.8642869733826899, 0.8494076099339257, 0.15059239006607428, 0.13571302661731008, 0.7362065926317646, 0.7919799498746868, 0.0696969696969697, 0.06948704389212057, 0.06948704389212057, 0.020548546858855272, 0.0198145163666447], 'Logistic regression', 'Default without balancer'), ([0.69320035419795056, 0.69173555228400818, 0.8435832321537461, 0.7958928256654327, 0.8912736386420597, 0.1087263613579403, 0.20410717433456724, 0.6190551736294729, 0.7162793347003872, 0.03915470494417863, 0.06409307244843998, 0.06409307244843998, 0.25462352114789066, 0.021031406219650328], 'Random forest', 'Default without balancer'), ([0.72069002449359998, 0.71167906044612561, 0.8621282576423271, 0.926158998766085, 0.7980975165185692, 0.20190248348143083, 0.07384100123391503, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08865309562426138, 0.028387494107142516], 'SVM (RDF)', 'Default without balancer'), ([0.72069002449359998, 0.71167906044612561, 0.8621282576423271, 0.926158998766085, 0.7980975165185692, 0.20190248348143083, 0.07384100123391503, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10728402362279428, 0.028387494107142516], 'SVM (linear)', 'Default without balancer'), ([0.72069002449359998, 0.71167906044612561, 0.8621282576423271, 0.926158998766085, 0.7980975165185692, 0.20190248348143083, 0.07384100123391503, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09565213279002445, 0.028387494107142516], 'SVM (polynomial)', 'Default without balancer')], 'Default without balancer'), ('Default with balancer', [([0.68564460446134967, 0.6848019807991047, 0.8442853082420005, 0.8512956107879428, 0.8372750056960583, 0.1627249943039417, 0.14870438921205711, 0.0195134849286092, 0.0008658008658008658, 0.0, 0.0, 0.0, 1.0859752915141891, 0.020431428192996332], 'AdaBoost', 'Default with balancer'), ([0.70228689550148016, 0.70138224062994858, 0.852613662419112, 0.861025912215759, 0.8442014126224654, 0.15579858737753474, 0.13897408778424114, 0.7536400493566014, 0.7875825928457507, 0.07665755297334245, 0.0824960338445267, 0.0824960338445267, 3.3051575776986284, 0.013774492104555905], 'Artificial neural network', 'Default with balancer'), ([0.69963350749755115, 0.69957615551701646, 0.8495731902228046, 0.8305658381808567, 0.8685805422647528, 0.13141945773524719, 0.1694341618191433, 0.7796404019037547, 0.8084529505582138, 0.08442697653223968, 0.10967741935483871, 0.10967741935483871, 0.05185550100937064, 0.015135379860056435], 'Bernoulli Naive Bayes', 'Default with balancer'), ([0.69317910261610216, 0.67484466678008104, 0.8461174841097745, 0.9446500969504671, 0.7475848712690819, 0.25241512873091815, 0.05534990304953288, 0.9446500969504671, 0.7475848712690819, 0.25241512873091815, 0.05534990304953288, 0.05534990304953288, 0.011421795955009506, 0.028170032674156548], 'Decision Tree', 'Default with balancer'), ([0.6837963833258941, 0.6779898946760089, 0.8438989702996494, 0.8913978494623654, 0.7964000911369333, 0.20359990886306678, 0.1086021505376344, 0.7232152300370175, 0.6013442697653224, 0.12359307359307359, 0.05540278512251013, 0.05540278512251013, 0.052062476662413225, 0.028303761008582295], 'Extreme Learning Machine', 'Default with balancer'), ([0.68733972901560192, 0.67655017138328688, 0.8447956039803851, 0.9140842587696104, 0.7755069491911598, 0.2244930508088403, 0.08591574123038954, 0.912991362594747, 0.7737525632262475, 0.2244930508088403, 0.08591574123038954, 0.08591574123038954, 0.00541577907115028, 0.02692255849844871], 'Gaussian Naive Bayes', 'Default with balancer'), ([0.66901481900157922, 0.66823669326872981, 0.8358403369849555, 0.8413890357835361, 0.8302916381863752, 0.16970836181362495, 0.15861096421646395, 0.7264410364886303, 0.6962292093871042, 0.08091820460241513, 0.10648686761854398, 0.10648686761854398, 0.006196490831944364, 0.024343757236329228], 'K-nearest neighbours', 'Default with balancer'), ([0.72342885326694795, 0.72090745493011621, 0.8637874010539545, 0.8903225806451612, 0.8372522214627477, 0.16274777853725222, 0.10967741935483871, 0.7741935483870966, 0.7798017771701983, 0.09228753702437913, 0.0694694165344615, 0.0694694165344615, 0.01992731803927091, 0.0201642135671188], 'Logistic regression', 'Default with balancer'), ([0.71111374491701496, 0.71087538996550836, 0.8547207612078608, 0.8295258240789706, 0.879915698336751, 0.12008430166324902, 0.17047417592102945, 0.6842411422527762, 0.6970494417862838, 0.05225563909774436, 0.05751806804160057, 0.05751806804160057, 0.2706357090656836, 0.02040343312322797], 'Random forest', 'Default with balancer'), ([0.72069002449359998, 0.71167906044612561, 0.8621282576423271, 0.926158998766085, 0.7980975165185692, 0.20190248348143083, 0.07384100123391503, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08316708368688752, 0.028387494107142516], 'SVM (RDF)', 'Default with balancer'), ([0.71794275204220115, 0.70808221639866586, 0.8606004278038011, 0.9283095364004937, 0.7928913192071088, 0.20710868079289133, 0.07169046359950644, 0.0, 0.0, 0.0, 0.0, 0.0, 0.046385487676194444, 0.028709441839245932], 'SVM (linear)', 'Default with balancer'), ([0.72125644168289027, 0.71180111771044785, 0.8623377255937306, 0.9283095364004937, 0.7963659147869674, 0.20363408521303258, 0.07169046359950644, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05130963668377336, 0.028441462428769493], 'SVM (polynomial)', 'Default with balancer')], 'Default with balancer'), ('Tuned', [([0.6999852300658439, 0.69948664951841444, 0.8482675729273185, 0.8131676361713379, 0.8833675096832992, 0.11663249031670085, 0.18683236382866208, 0.33467301251542386, 0.3993620414673047, 0.009569377990430622, 0.040190375462718135, 0.040190375462718135, 0.20060690050991273, 0.019391473182383462], 'AdaBoost', 'Tuned'), ([0.71389031553003546, 0.71298168247397031, 0.8583409511712553, 0.8664198836594394, 0.8502620186830713, 0.1497379813169287, 0.13358011634056055, 0.7481755684822845, 0.7989405331510596, 0.07664616085668717, 0.08469945355191255, 0.08469945355191255, 9.065388609617656, 0.02665056341162305], 'Artificial neural network', 'Tuned'), ([0.70069975048977318, 0.70064942280681952, 0.8502243722859381, 0.8327340031729245, 0.8677147413989519, 0.13228525860104806, 0.16726599682707563, 0.7796404019037547, 0.8101959444064709, 0.08615857826384142, 0.1096774193548387, 0.1096774193548387, 0.018129235919745135, 0.01598496276885547], 'Bernoulli Naive Bayes', 'Tuned'), ([0.63866969594170364, 0.63656400219393416, 0.8210358596843053, 0.8415476820024678, 0.8005240373661425, 0.19947596263385736, 0.15845231799753215, 0.808901815617839, 0.7215083162451584, 0.15600364547732967, 0.10417768376520359, 0.10417768376520359, 0.0058035691376665, 0.04120838909119729], 'Decision Tree', 'Tuned'), ([0.72236136212956659, 0.71354050067916985, 0.8630085682452852, 0.926176626123744, 0.7998405103668261, 0.20015948963317387, 0.07382337387625594, 0.7144896879957695, 0.7745841877420826, 0.06791979949874687, 0.05646042658205535, 0.05646042658205535, 0.10459152897050383, 0.028319200116898747], 'Extreme Learning Machine', 'Tuned'), ([0.68733972901560192, 0.67655017138328688, 0.8447956039803851, 0.9140842587696104, 0.7755069491911598, 0.2244930508088403, 0.08591574123038954, 0.912991362594747, 0.7737525632262475, 0.2244930508088403, 0.08591574123038954, 0.08591574123038954, 0.0055341083701169564, 0.02692255849844871], 'Gaussian Naive Bayes', 'Tuned'), ([0.71828141663033873, 0.71391958219365048, 0.8527050644254017, 0.7828309536400493, 0.9225791752107542, 0.07742082478924585, 0.21716904635995063, 0.5375991538868323, 0.7458304853041696, 0.025233538391433125, 0.07278335977436982, 0.07278335977436982, 0.004802801104044481, 0.019848661451726014], 'K-nearest neighbours', 'Tuned'), ([0.71038031649857813, 0.70752091996987565, 0.8573032413975663, 0.8860920148069805, 0.8285144679881523, 0.1714855320118478, 0.11390798519301955, 0.7764498501674599, 0.778047391205286, 0.10804283435862383, 0.06839414771725717, 0.06839414771725717, 0.026088491197799227, 0.019638156656227042], 'Logistic regression', 'Tuned'), ([0.72034042091669936, 0.72028498944596142, 0.8601422580777697, 0.8447206063811034, 0.875563909774436, 0.1244360902255639, 0.15527939361889653, 0.6363652388506963, 0.7580770107085897, 0.03484848484848486, 0.05644279922439627, 0.05644279922439627, 2.684516581043948, 0.019586961752419457], 'Random forest', 'Tuned'), ([0.72748036801887883, 0.72279233807714094, 0.8659525839742378, 0.9076855279393619, 0.8242196400091139, 0.1757803599908863, 0.0923144720606381, 0.8340384276396967, 0.7911255411255412, 0.11146046935520619, 0.07274810505905165, 0.07274810505905165, 0.5421912875626345, 0.029935482678179203], 'SVM (RDF)', 'Tuned'), ([0.7238632849616794, 0.71309340407746424, 0.8634436374576652, 0.934884540807333, 0.7920027341079973, 0.2079972658920027, 0.06511545919266702, 0.9240084611316765, 0.7920027341079973, 0.2062542720437457, 0.06402256301780362, 0.06402256301780362, 0.6789859943791569, 0.027798374571698665], 'SVM (linear)', 'Tuned'), ([0.61289088207539411, 0.59014038408292302, 0.7866578403366234, 0.6343204653622422, 0.9389952153110048, 0.0610047846889952, 0.36567953463775776, 0.7025559668605678, 0.7309979494190021, 0.07402597402597402, 0.07821258593336859, 0.07821258593336859, 0.5626958634149855, 0.028302437342853547], 'SVM (polynomial)', 'Tuned')], 'Tuned')]]
    plot_percentage_difference_graph(results, ["Lima TB", "India Attrition", "German Credit", "Australian Credit"], x_label="Parameter tuning approach", name_suffix="", difference_from="using default parameters", figsize=(16, 5), legend_y=-0.62, label_rotation=0, y_label_pos=-0.3)