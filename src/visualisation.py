import os
from collections import OrderedDict
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
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
import config.visualisation_input as vis_input

import constants as const

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

def plot_percentage_difference_graph(results, datasets, name_suffix="", parameter="BARR", x_label="Feature selection approach", difference_from="no feature selection", figsize=(16, 5), legend_y=-0.31, label_rotation=0, y_label_pos=-0.4, y_ticks=None, x_label_replacement_dict=None, feature_selection_specific=False):
    if x_label_replacement_dict is None:
        x_label_replacement_dict = {}

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

        feature_selection_labels = [results[0][i][0] if results[0][i][0] not in x_label_replacement_dict else x_label_replacement_dict[results[0][i][0]] for i in range(1, len(results[0]))]
        if feature_selection_specific:
            feature_selection_labels = [feature_selection_labels[i-1] + "\n{0}-{1:.1f}-{2}".format(results[0][i][1][0][4][0], results[0][i][1][0][4][1], results[0][i][1][0][4][2]) for i in range(1, len(results[0]))]

        plt.xticks(data_balancers + (bar_width / 2) * len(classifiers), feature_selection_labels, rotation=label_rotation)
        bonus = ""
        if feature_selection_specific:
            bonus = " ({0})".format(results[0][0][1][0][4][3])
        plt.title(datasets[0].replace("_", " ") + bonus)
        plt.ylabel("Change in {0} from {1}".format(parameter, difference_from), y=y_label_pos)

    vertical_plt = 0
    for z in range(1, len(results)):
        ax2 = plt.subplot(subplt_val + z, sharey=ax1)
        color = iter(cm.Set1(np.linspace(0, 1, len(no_feature_selection) + 1)))
        for i in range(len(classifier_arr[z])):
            if i + 1 != len(classifier_arr[z]):
                label = results[z][0][1][i][1]
            else:
                label = "Mean classification"
            data_balancers = np.arange(len(classifier_arr[z][i])) * 3
            plt.bar(data_balancers + (i * bar_width), classifier_arr[z][i], bar_width,
                    alpha=opacity,
                    color=colors[i],
                    hatch=patterns[i % len(patterns)],
                    label=label)

        feature_selection_labels = [results[0][i][0] if results[0][i][0] not in x_label_replacement_dict else x_label_replacement_dict[results[0][i][0]] for i in range(1, len(results[0]))]
        if feature_selection_specific:
            feature_selection_labels = [feature_selection_labels[i-1] + "\n{0}-{1:.1f}-{2}".format(results[z][i][1][0][4][0], results[z][i][1][0][4][1], results[z][i][1][0][4][2]) for i in range(1, len(results[0]))]

        plt.xticks(data_balancers + (bar_width / 2) * len(classifiers), feature_selection_labels, rotation=label_rotation)
        bonus = ""
        if feature_selection_specific:
            bonus = " ({0})".format(results[z][0][1][0][4][3])
        plt.title(datasets[z].replace("_", " ") + bonus)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    legend = plt.legend(loc='lower center', bbox_to_anchor=(-0.08, legend_y), fancybox=True, frameon=True, ncol=7)
    legend.get_frame().set_facecolor('#ffffff')

    if y_ticks is not None:
        plt.yticks(y_ticks)
        plt.ylim(ymin=y_ticks[0])
        plt.ylim(ymax=y_ticks[-1])
    plt.xlabel(x_label, x=0, y=-2)
    feature_selection_labels = [results[0][i][0] for i in range(1, len(results[0]))]


    plt.locator_params(axis='y', nbins=15)
    name = "{3}_results_per_classifier_plot{0}_{4}_{1}_{2}".format(name_suffix, parameter, current_time, x_label, datasets)
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/{0}".format(name.replace(" ", "_")), bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)


def plot_time_to_default_results(time_to_default_results_per_classifier, parameter="Change in balanced accuracy from no feature selection"):
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
    plt.xlabel("Change in TPR")
    plt.ylabel("Change in TNR")

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


def visualise_dataset_balancer_results(results, range=(-0.5, 0.5), colors=("#64B3DE", "#1f78b4", "#B9B914", "#FBAC44", "#bc1659", "#33a02c", "grey" , "#b15928", "#6a3d9a", "#e31a1c", "#6ABF20", "#ff7f00", "#6a3d9a"), exclude=("SVM (linear)", "Logistic regression", "Random forest")):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = "raw_dump_{0}.txt".format(current_time)
    with open(os.path.dirname(os.path.realpath(__file__)) + "/../results/" + file_name, "wb") as output_file:
        output_file.write(str(results))
    sns.set(style='ticks')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    markers = ["s", "d", "o", "^", "*"]
    size = [150, 200, 200, 200, 250]
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
    plt.xlabel("Change in TPR")
    plt.ylabel("Change in TNR")

    ax.xaxis.set_label_coords(0.1, 0.53)
    ax.yaxis.set_label_coords(0.53, 0.9)

    plt.ylim(range[0], range[1])
    plt.xlim(range[0], range[1])
    balancer_labels = ([], [])
    classifier_labels = ([], [])
    data_set_index = 0
    for (data_set, dataset_result) in results:

        none_true_pos_per_classifier = {}
        none_true_neg_per_classifier = {}

        for (classifier_description, result_arr) in dataset_result:
            for (balancer_description, results) in result_arr:
                if balancer_description == "None":
                    none_true_pos_per_classifier[classifier_description] = results[3]
                    none_true_neg_per_classifier[classifier_description] = results[4]
                    break

        i = 0
        for (classifier_description, result_arr) in dataset_result:
            if classifier_description in exclude:
                continue
            balancer_index = 0
            for (balancer_description, results) in result_arr:
                if balancer_description != "None":
                    if data_set_index == 0 and balancer_index == 0:
                        classifier_labels[0].append(mpatches.Patch(color=colors[i], label=classifier_description, alpha=0.8))
                        classifier_labels[1].append(classifier_description)
                    ax.scatter(results[3] -  none_true_pos_per_classifier[classifier_description], results[4] - none_true_neg_per_classifier[classifier_description], marker=markers[balancer_index % len(markers)], hatch=hatches[balancer_index % len(hatches)], s=size[balancer_index % len(markers)], alpha=0.8, color=colors[i], edgecolor="black" if colors[i] != "black" else "grey", zorder=balancer_index % len(markers), lw=0.8)
                    pt = ax.scatter(-99999999999, -9999999999, marker=markers[balancer_index % len(markers)], hatch=hatches[balancer_index % len(hatches)], s=200, alpha=0.8, color="white", edgecolor="black", zorder=data_set_index, lw=0.8)
                    if i == 0:
                        balancer_labels[0].append(pt)
                        balancer_labels[1].append(balancer_description)
                    balancer_index += 1
            i += 1
        data_set_index += 1
    legend = plt.legend(balancer_labels[0] + classifier_labels[0], balancer_labels[1] + classifier_labels[1], loc='lower center', bbox_to_anchor=(0.5, -0.2), fancybox=False, frameon=False, ncol=7)
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
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    markers = ["s", "o", "^", "d", "*"]
    sizes = [150, 200, 200, 200, 250]
    colors = ["#64B3DE", "#1f78b4", "#B9B914", "#FBAC44", "#bc1659", "#33a02c", "#6ABF20", "#ff7f00", "#6a3d9a", "grey", "#b15928", "#e31a1c", "black"]
    hatches = [None, "////", ".."]
    color_dict = {}
    index = 0
    for (classifier_description, result_arr) in dataset_results[0][1]:
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
    plt.xlabel("Change in TPR")
    plt.ylabel("Change in TNR")

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

        i = 0
        hatch_index = 0
        for key, value in balancer_result_pos.iteritems():
            if key != "None":
                if i != 0 and hatch_index == 0 and i % len(colors) == 0:
                    hatch_index += 1

                if data_set_index == 0:
                    balancer_labels[0].append(mpatches.Patch(facecolor=colors[i % len(colors)], hatch=hatches[hatch_index], label=key, alpha=0.8, edgecolor="black"))
                    balancer_labels[1].append(key)

                ax.scatter(value - balancer_result_pos["None"], balancer_result_neg[key] - balancer_result_neg["None"], marker=markers[data_set_index % len(markers)], hatch=hatches[hatch_index], s=sizes[data_set_index % len(markers)], alpha=0.8, color=colors[i % len(colors)], edgecolor="black" if colors[i % len(colors)] != "black" else "grey", zorder=i % len(markers), lw=0.8)
                pt = ax.scatter(-99999999999, -9999999999, marker=markers[data_set_index % len(markers)], s=sizes[data_set_index % len(markers)], alpha=0.8, color="white", edgecolor="black", zorder=data_set_index, lw=0.8)
                if i == 0:
                    data_set_labels[0].append(pt)
                    data_set_labels[1].append(data_set)
                i += 1
                hatch_index = (hatch_index + 1) % len(hatches)
        data_set_index += 1
    legend = plt.legend(data_set_labels[0] + balancer_labels[0], data_set_labels[1] + balancer_labels[1], loc='upper right', bbox_to_anchor=(1, 1), fancybox=False, frameon=False, ncol=1)
    legend.get_frame().set_facecolor('#ffffff')

    sns.despine()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/classifier_dataset_plt_{0}.png".format(current_time), bbox_extra_artists=((legend,)), bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    plot_percentage_difference_graph(vis_input.results, ["Lima TB", "India Attrition", "German Credit", "Australia Credit"], x_label="Feature selection approach", name_suffix="_after", difference_from="no feature selection", figsize=(20, 4.5), legend_y=-0.79, x_label_replacement_dict={"Logistic regression": "LR", "Decision Tree": "DT", "Bernoulli Naive Bayes": "Bernoulli NB", "Random forest": "RF"}, feature_selection_specific=True, y_ticks=np.arange(-0.4, 0.11, 0.03))
#result_arr, dataset_arr, x_label="Feature selection approach", name_suffix="_after", difference_from="no feature selection", figsize=(16, 4.5), legend_y=-0.79, label_rotation=0, x_label_replacement_dict={"Logistic regression": "LR", "Decision Tree": "DT", "Bernoulli Naive Bayes": "Bernoulli NB", "Random forest": "RF"}