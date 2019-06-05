import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from .SentiDep import SentiDep
from tqdm import tqdm


class SentiPlotting:

    def __init__(self):
        """
        Plotting class for normed plotting.
        Only three plots available: heatmap, barplot, boxplot
        """
        self.palette = ["#294252", "#b30033", "#DA8B88", "#696766", "#8D8889", "#B1ACAD",
                        "#d8cac1", "#EBE4E1", "#42637f", "#6D869F", "#A4B2C5"]

    def heatmap(self, pandas_dataframe, **kwargs):
        """
        Creates a heatmap using a pandas-DataFrame as input.
        :param pandas_dataframe: DataFrame for plotting
        :param kwargs: number_of_observations, figure_save_path,
        vmax, vmin, title, xlabel, ylabel
        :return: seaborn/plt object
        """
        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        nobs = kwargs.get('number_of_observations', None)
        save_path = kwargs.get('figure_save_path', None)
        vmax = kwargs.get('vmax', None)
        vmin = kwargs.get('vmin', None)
        title = kwargs.get('title', "Heatmap")
        xlabel = kwargs.get('xlabel', None)
        ylabel = kwargs.get('ylabel', None)

        fig = plt.figure()
        plotted_df = sns.heatmap(data=pandas_dataframe, cmap='magma',
                                 annot=True, vmax=vmax, vmin=vmin)
        plotted_df.set_title(title, fontsize=22)
        if xlabel:
            plotted_df.set_xlabel(xlabel, fontsize=18)
        if ylabel:
            plotted_df.set_ylabel(ylabel, fontsize=18)
        xticklabels = [l.get_text() for l in plotted_df.get_xticklabels()]
        if nobs:
            xticklabels = [l + "\nn = " + str(n) for l, n in zip(xticklabels, nobs)]
        plotted_df.set_xticklabels(xticklabels, fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        return plotted_df

    def barplot(self, pandas_dataframe, **kwargs):
        """
        Creates a barplot using a pandas-DataFrame as input.
        :param pandas_dataframe: DataFrame for plotting
        :param kwargs: number_of_observations, figure_save_path,
        xlim -> as tuple(low, high), ylim -> as tuple(low, high),
        title, xlabel, ylabel
        :return: seaborn/plt object
        """
        title = kwargs.get('title', "Boxplot")
        xlabel = kwargs.get('xlabel', None)
        ylabel = kwargs.get('ylabel', None)
        nobs = kwargs.get('number_of_observations', None)
        save_path = kwargs.get('figure_save_path', None)
        ylim = kwargs.get('ylim', None)  # as tuple(low, high)
        xlim = kwargs.get('xlim', None)  # as tuple(low, high)

        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        sns.set_style("darkgrid", {"axes.facecolor": "#EBE4E1"})
        fig = plt.figure()
        plotted_df = sns.barplot(data=pandas_dataframe, saturation=1,
                                 palette=sns.color_palette(self.palette))
        plotted_df.set_title(title, fontsize=22)
        if xlabel:
            plotted_df.set_xlabel(xlabel, fontsize=18)
        if ylabel:
            plotted_df.set_ylabel(ylabel, fontsize=18)
        xticklabels = [l.get_text() for l in plotted_df.get_xticklabels()]
        if nobs:
            for i, p in enumerate(plotted_df.patches):
                height = p.get_height()
                plotted_df.text(p.get_x() + p.get_width() / 2.,
                                height + 3, "n = " + str(nobs[i]), ha="center", fontsize=14)
        plotted_df.set_xticklabels(xticklabels, fontsize=14, rotation=90)
        plt.yticks(fontsize=14)
        if ylim:
            plt.ylim(ylim[0], ylim[1])
        if xlim:
            plt.xlim(xlim[0], xlim[1])
        plt.show()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        return plotted_df

    def boxplot(self, pandas_dataframe, **kwargs):
        """
        Creates a boxplot using a pandas-DataFrame as input.
        :param pandas_dataframe: DataFrame for plotting
        :param kwargs: number_of_observations, figure_save_path,
        xlim -> as tuple(low, high), ylim -> as tuple(low, high),
        title, xlabel, ylabel
        :return: seaborn/plt object
        """
        title = kwargs.get('title', "Boxplot")
        xlabel = kwargs.get('xlabel', None)
        ylabel = kwargs.get('ylabel', None)
        nobs = kwargs.get('number_of_observations', None)
        ylim = kwargs.get('ylim', None)     # as tuple(low, high)
        xlim = kwargs.get('xlim', None)     # as tuple(low, high)
        save_path = kwargs.get('figure_save_path', None)

        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        sns.set_style("darkgrid", {"axes.facecolor": "#EBE4E1"})
        fig = plt.figure()
        plotted_df = sns.boxplot(data=pandas_dataframe, saturation=1,
                                 palette=sns.color_palette(["#6D869F"]))
        plotted_df.set_title(title, fontsize=22)
        if xlabel:
            plotted_df.set_xlabel(xlabel, fontsize=18)
        if ylabel:
            plotted_df.set_ylabel(ylabel, fontsize=18)
        xticklabels = [l.get_text() for l in plotted_df.get_xticklabels()]
        if nobs:
            xticklabels = [l + "\nn = " + str(n) for l, n in zip(xticklabels, nobs)]
        plotted_df.set_xticklabels(xticklabels, fontsize=14, rotation=90)
        plt.yticks(fontsize=14)
        if ylim:
            plt.ylim(ylim[0], ylim[1])
        if xlim:
            plt.xlim(xlim[0], xlim[1])
        plt.show()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        return plotted_df

    def polarityplot_lickert(self, intervals_df, **kwargs):
        """
        Plots categorized polarity-values as lickert-plot
        :param intervals_df: output from 'prepare_for_polarityplot'-function in form
                             columns: five intervals from 'sehr schlecht' to 'sehr gut'
                             rows: keywords/topics
                             values: number of observations for each category
        :param kwargs: title = str(set title to a string of your choice)
                       figure_save_path = str(give a path to save the plot in)
                       -> not given: plot-figure will be dropped after executing
        :return: matplotlib ax
        """
        title = kwargs.get('title', "Sentimentpolaritäten als Lickert-plot")
        save_path = kwargs.get('figure_save_path', None)
        likert_colors = ['white', 'firebrick', 'lightcoral', 'gainsboro', 'cornflowerblue', 'darkblue']
        middles = intervals_df[["sehr schlecht", "schlecht"]].sum(axis=1) + intervals_df["neutral"] * .5
        longest = middles.max()
        intervals_df.insert(0, '', (middles - longest).abs())
        ax = intervals_df.plot.barh(stacked=True, color=likert_colors, edgecolor='none')
        ax.legend(loc=6, bbox_to_anchor=(1, 0.5), fontsize=14)
        z = plt.axvline(longest, linestyle='--', color='black', alpha=.5)
        z.set_zorder(-1)
        ax.set_title(title, fontsize=22)
        ax.set_xlabel("Anteil der Polaritäten in %", fontsize=18)
        ax.set_ylabel("Themen/Schlüsselbegriffe", fontsize=18)
        ax.figure.set_size_inches(12, 8)
        xvalues = range(-100 + int(round(longest)), 110 + int(round(longest)), 10)
        xlabels = [str(x - int(round(longest))) for x in xvalues]
        plt.xticks(xvalues, xlabels, fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()
        if save_path:
            ax.figure.savefig(save_path, bbox_inches='tight')
        return ax

    def prepare_for_polarityplot(self, polarity_df, **kwargs):
        """
        Prepares a formatted dataframe from 'SentiDep.create_polarity_df'
        for plotting via the 'polarityplot_lickert'-function
        :param polarity_df: formatted dataframe in form:
                            columns: keywords/topics
                            rows: document-keys
                            values: float(polarity-scores) or np.nan
        :param kwargs: intervals = list of five intervallimits for
                       polarity classification in form:
                       [-1.0, -0.6, 0.2, 0.6, 1.0] <- default
                       (middlescore (here: 0.2) is mirrored on 0
                       -> symmetric neutral range -> I=[-0.2;0.2])
        :return: categorized dataframe in form:
                 columns: five intervals from 'sehr schlecht' to 'sehr gut'
                 rows: keywords/topics
                 values: number of observations for each category
        """
        intervals = {"sehr schlecht": -1.0, "schlecht": -0.6, "neutral": 0.2, "gut": 0.6, "sehr gut": 1.0}
        iv = kwargs.get('intervals', intervals)
        aggregate = kwargs.get('aggregate_dict', None)
        if type(iv) == list:
            new_intervals = {}
            for key, value in zip(intervals.keys(), iv):
                new_intervals[key] = value
            intervals = new_intervals
        if aggregate:
            agg_list = aggregate.items()
            max_len = len(list(agg_list))
            intervals_dict = {"sehr schlecht": [0] * max_len, "schlecht": [0] * max_len,
                                   "neutral": [0] * max_len, "gut": [0] * max_len,
                                   "sehr gut": [0] * max_len}
            nobs = {}
            counter = {}
            for c in tqdm(polarity_df.columns):
                c_scores = polarity_df[c].values.tolist()
                for key, value in agg_list:
                    if not key in nobs.keys():
                        nobs[key] = 0
                    if not key in counter.keys():
                        counter[key] = {"sehr schlecht": 0, "schlecht": 0,
                                               "neutral": 0, "gut": 0,
                                               "sehr gut": 0}
                    if c in value:
                        nob = len(list(filter(lambda x: not np.isnan(x), c_scores)))
                        nobs[key] += nob
                        # counter = {"sehr schlecht": 0, "schlecht": 0, "neutral": 0, "gut": 0, "sehr gut": 0}
                        for cs in c_scores:
                            if intervals["sehr schlecht"] <= cs < intervals["schlecht"] or \
                                    cs < intervals["sehr schlecht"]:
                                counter[key]["sehr schlecht"] += 1
                            elif intervals["schlecht"] <= cs < -intervals["neutral"]:
                                counter[key]["schlecht"] += 1
                            elif -intervals["neutral"] <= cs <= intervals["neutral"]:
                                counter[key]["neutral"] += 1
                            elif intervals["neutral"] < cs <= intervals["gut"]:
                                counter[key]["gut"] += 1
                            elif intervals["gut"] < cs <= intervals["sehr gut"] or \
                                    cs > intervals["sehr gut"]:
                                counter[key]["sehr gut"] += 1
            for idx, (group, count) in enumerate(counter.items()):
                for cat, ct in count.items():
                    if nobs[group] > 0:
                        intervals_dict[cat][idx] += round(ct / nobs[group] * 100)

            interval_ids = ["{}: n = {}".format(key, value) for key, value in nobs.items()]
            intervals_df = pd.DataFrame(data=intervals_dict, index=interval_ids)
            return intervals_df

        else:
            intervals_dict = {"sehr schlecht": [], "schlecht": [], "neutral": [], "gut": [], "sehr gut": []}
            nobs = {}
            for c in tqdm(polarity_df.columns):
                c_scores = polarity_df[c].values.tolist()
                nobs[c] = len(list(filter(lambda x: not np.isnan(x), c_scores)))
                counter = {"sehr schlecht": 0, "schlecht": 0, "neutral": 0, "gut": 0, "sehr gut": 0}
                for cs in c_scores:
                    if intervals["sehr schlecht"] <= cs < intervals["schlecht"] or \
                            cs < intervals["sehr schlecht"]:
                        counter["sehr schlecht"] += 1
                    elif intervals["schlecht"] <= cs < -intervals["neutral"]:
                        counter["schlecht"] += 1
                    elif -intervals["neutral"] <= cs <= intervals["neutral"]:
                        counter["neutral"] += 1
                    elif intervals["neutral"] < cs <= intervals["gut"]:
                        counter["gut"] += 1
                    elif intervals["gut"] < cs <= intervals["sehr gut"] or \
                            cs > intervals["sehr gut"]:
                        counter["sehr gut"] += 1
                for cat, count in counter.items():
                    if nobs[c] > 0:
                        intervals_dict[cat].append(round(count / nobs[c] * 100))
                    else:
                        intervals_dict[cat].append(0)
            interval_ids = ["{}: n = {}".format(key, value) for key, value in nobs.items()]
            intervals_df = pd.DataFrame(data=intervals_dict, index=interval_ids)
            return intervals_df

    '''
        def horizontal_polarityplot(self, dataframe, **kwargs):
            """
            Creates a horizontal barplot with polarity scores on x-axis
            and keywords on y-axis.
            :param dataframe: keywords as columns, doc as row
            polarity-scores as values
            :param kwargs: figure_save_path, title,
            nobs -> as dict{keyword1: nobs1, ...},
            intervals: -> intervals of classification
                       -> list of 5 floats between -1 and 1
                       -> float with index 3 is positive and
                          describes the neutral area around +- 0
                       -> default: [-1.0, -0.75, 0.25, 0.75, 1.0]
            :return: ax of plot
            """
            save_path = kwargs.get('figure_save_path', None)
            title = kwargs.get('title', None)
            nobs = kwargs.get('nobs', None)
            iv = [-1, -0.75, 0.25, 0.75, 1]
            num_iv = kwargs.get('intervals', iv)
            intervals = {num_iv[i]: [] for i in range(len(num_iv))}
            means = dataframe.mean().values.tolist()
            stds = dataframe.std().tolist()
            for m in means:
                if m == 0:
                    for inter_key in intervals.keys():
                        if inter_key == num_iv[round(len(num_iv) / 2)]:
                            intervals[inter_key].append(m)
                        else:
                            intervals[inter_key].append(np.nan)
                elif m > 0:
                    if m <= num_iv[2]:
                        intervals[num_iv[2]].append(m)
                        intervals[num_iv[0]].append(np.nan)
                        intervals[num_iv[1]].append(np.nan)
                        intervals[num_iv[3]].append(np.nan)
                        intervals[num_iv[4]].append(np.nan)
                    elif m <= num_iv[3]:
                        intervals[num_iv[3]].append(m)
                        intervals[num_iv[0]].append(np.nan)
                        intervals[num_iv[1]].append(np.nan)
                        intervals[num_iv[2]].append(num_iv[2])
                        intervals[num_iv[4]].append(np.nan)
                    elif m <= num_iv[4]:
                        intervals[num_iv[4]].append(m)
                        intervals[num_iv[0]].append(np.nan)
                        intervals[num_iv[1]].append(np.nan)
                        intervals[num_iv[2]].append(num_iv[2])
                        intervals[num_iv[3]].append(num_iv[3])
                elif m < 0:
                    if m >= -num_iv[2]:
                        intervals[num_iv[2]].append(m)
                        intervals[num_iv[0]].append(np.nan)
                        intervals[num_iv[1]].append(np.nan)
                        intervals[num_iv[3]].append(np.nan)
                        intervals[num_iv[4]].append(np.nan)
                    elif m >= num_iv[1]:
                        intervals[num_iv[1]].append(m)
                        intervals[num_iv[0]].append(np.nan)
                        intervals[num_iv[3]].append(np.nan)
                        intervals[num_iv[2]].append(-num_iv[2])
                        intervals[num_iv[4]].append(np.nan)
                    elif m >= num_iv[0]:
                        intervals[num_iv[0]].append(m)
                        intervals[num_iv[4]].append(np.nan)
                        intervals[num_iv[1]].append(num_iv[1])
                        intervals[num_iv[2]].append(-num_iv[2])
                        intervals[num_iv[3]].append(np.nan)
            print(intervals)
            mean_df = pd.DataFrame(data=intervals,
                                   index=dataframe.columns).reset_index()

            sns.set(style="darkgrid")
            f, ax = plt.subplots(figsize=(16, 12))
            palette = ["#FF0000", "#FE642E", "#BDBDBD", "#00BFFF", "#0080FF"]
            sns.barplot(x=num_iv[4], y='index', data=mean_df,
                        palette=sns.color_palette([palette[4]]), label="sehr gut")
            sns.barplot(x=num_iv[3], y='index', data=mean_df,
                        palette=sns.color_palette([palette[3]]), label="gut")
            sns.barplot(x=num_iv[0], y='index', data=mean_df,
                        palette=sns.color_palette([palette[0]]), label="sehr schlecht")
            sns.barplot(x=num_iv[1], y='index', data=mean_df,
                        palette=sns.color_palette([palette[1]]), label="schlecht")
            sns.barplot(x=num_iv[2], y='index', data=mean_df,
                        palette=sns.color_palette([palette[2]]), label="neutral")
            ax.legend(ncol=1)
            if title:
                ax.set_title(title, fontsize=22)
            ax.set_ylabel("Schlüsselwort", fontsize=18)
            ax.set_xlabel("Polarität nach SentiWS", fontsize=18)
            ax.set_xlim(-1, 1)
            if nobs:
                ax.set_yticklabels([yt.get_text() + ": n = " + str(ld) for ld, yt in \
                                    zip(nobs.values(), ax.get_yticklabels())])
            [xt.set_fontsize(14) for xt in ax.get_xticklabels()]
            [yt.set_fontsize(14) for yt in ax.get_yticklabels()]
            if save_path:
                f.savefig(save_path, bbox_inches='tight')
            plt.show()
            return ax
        '''


if __name__ == '__main__':
    single_ratings = pd.read_excel("../Klinikbewertungen/Klinikbewertung_Einzelbewertungen.xlsx")
    single_reports = single_ratings["Erfahrungsbericht"].values.tolist()
    sd = SentiDep(sentiws_file="../sentiws.pickle", polarity_modifiers_file="../polarity_modifiers.pickle",
                  negations_file="../negationen_lexicon.pickle", stts_file="../stts.pickle")
    topics = sd.generate_topics("".join(single_reports[:50]))
    topics.plot(30)

    def max_sentiment(topics_dict):
        return len(max(topics_dict.items(), key=lambda x: len(x[1]))[1])

    first_30 = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:30]
    first_30 = [fs[0] for fs in first_30]
    print(first_30)
    polarities = []
    for sr in single_reports[:10]:
        polarities.append(sd.get_depending_polarities(text=sr, keywords=first_30))
    topics_dict = {}
    for pol in polarities:
        for p in pol:
            if p:
                if not p[0] in topics_dict.keys():
                    topics_dict[p[0]] = []
                topics_dict[p[0]].append((p[1], p[2], p[3]))
    norm_dict = {}
    ld = {}
    max_len = max_sentiment(topics_dict)
    for term, value in topics_dict.items():
        norm_dict[term] = [v[0] for v in value] + [np.nan] * (max_len - len(value))
        ld[term] = len(value)
    mean_dict = norm_dict
    len_dict = ld
    print("length of mean_dict.keys: ", len(mean_dict.keys()))
    print("length of len_dict.keys", len(len_dict.keys()))
    sentiment_df = pd.DataFrame(data=mean_dict)
    print(sentiment_df)
    sp = SentiPlotting()
    sp.horizontal_polarityplot(sentiment_df,
                            intervals=[-1, -0.5, 0.2, 0.5, 1],
                            title="Polaritäten der Erfahrungsberichte von first 10",
                            nobs=len_dict)
