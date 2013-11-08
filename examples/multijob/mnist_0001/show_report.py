import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os
import shutil

from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size("small")


def report(df, x, loc, ylim, line_groups, figure_groups):
    try:
        os.makedirs(loc)
    except OSError:
        # don't care if the dir exists
        pass
        
    y_keys = ["train_y_misclass", "test_y_misclass"]


    if figure_groups == []:
        figure_groups_iter = [("all", df)]
    else:
        figure_groups_iter = df.groupby(figure_groups)

    # create figures
    figure_group_names_lists = {}

    for figure_names, figure_group in figure_groups_iter:
        if isinstance(figure_names, (list, tuple)):
            figure_names = "-".join(map(str, figure_names))

        lo_x = df[x].min()
        hi_x = df[x].max()
        
        for y_key in y_keys:
            fig = plt.figure()

            colors = iter(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
            markers = iter(['o', '1', '2', '3', '4', 'v'])
            markers_colors = itertools.product(markers, colors)

            if line_groups == []:
                line_groups_iter = [("all", figure_group)]
            else:
                line_groups_iter = figure_group.groupby(line_groups)

            for name, line in line_groups_iter:
                marker, color = next(markers_colors)
                line = line.sort(x)
                line.plot(x=x, y=y_key, ax=fig.gca(), color=color, marker=marker, label=str(name))

            fig.gca().set_title(y_key + ": " + ", ".join(line_groups))
            fig.gca().legend(loc="lower right", prop=fontP)
            fig.gca().set_xlabel(x)
            fig.gca().set_xlim([lo_x, hi_x])
            fig.gca().set_ylim(ylim)

            fig_name_png = y_key + "_" + figure_names + ".png"
            fig.savefig(os.path.join(loc, fig_name_png))

            fig_name_pdf = y_key + "_" + figure_names + ".pdf"
            fig.savefig(os.path.join(loc, fig_name_pdf))
            
            figure_group_names_list = figure_group_names_lists.get(figure_names, [])
            figure_group_names_list.append(fig_name_png)
            figure_group_names_lists[figure_names] = figure_group_names_list

    # create html
    with open(os.path.join(loc, "index.html"), 'wb') as html_file:
        for group_name in sorted(figure_group_names_lists.keys()):
            fig_names = figure_group_names_lists[group_name]
            html_file.write("<h1>{}</h1></br>\n".format(group_name))
            for name in fig_names:
                html_file.write("<img src='{}'/>\n".format(name))
            html_file.write("<br/>\n\n")



if __name__ == "__main__":
    df = pd.read_csv("report.csv")

    zoom = [0, 0.2]
    
    report(df,
        loc="figures",
        x="params_prop",
        line_groups=["n_columns"],
        figure_groups=[], 
        ylim=zoom,
        )
