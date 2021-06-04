from math import exp
import numpy as np
import pickle
import glob
import os
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
from numpy.core.numeric import full
from numpy.lib.financial import _ipmt_dispatcher
import seaborn as sns
sns.set(palette="colorblind")
# sns.set()
# sns.color_palette("colorblind")
from pathlib import Path

from rbfdqn.plotting.learning_curve_plotter import get_scores, generate_plot
from rbfdqn.plotting.state_plotter import XYStatePlotter


blue, orange, green, red, purple, brown, pink, gray, yellow, sky = sns.color_palette('colorblind')

# I like purple, blue, green brown, red. Then orange last I guess.

dark_blue, dark_orange, dark_green, dark_red, dark_purple, dark_brown, dark_pink, dark_gray, dark_yellow, dark_sky = sns.color_palette('dark')


COLOR_MAP={
    "DOIE (Ours)": blue,
    "RND": green,
    # "MPE": brown,
    'MPE': gray,
    "OptBias": red,
    "RBFDQN": purple,
    # "Reward Shaping": purple,
    "Reward Shaping": dark_purple,
    # "Q-Shaping": green,
    "Q-Shaping": dark_brown,
    # "Anti Q-Shaping": gray,
    "Anti Q-Shaping": dark_yellow,
    "Uniform Optimism": blue,
    "Random Agent": sky,
    "OMEGA": brown,
}


def make_graphs(experiment_name,
                subdir,
                run_titles=None,
                smoothen=False,
                min_length=-1,
                only_longest=False,
                skip_failures=False,
                cumulative=False,
                all_seeds=False,
                show=True,
                big=False,
                min_seeds=1,
                max_episodes=-1,
                separate_titles=False,
                use_colormap=False,
                title=None,
                include_legend=True,
                linewidth=8,
                legend_loc="upper right"):
    if run_titles is None:
        print("Using all runs in experiment")
        run_titles = [
            subdir.basename for subdir in Path(experiment_name).iterdir()
            if subdir.is_dir()
        ]

    log_dirs = [
        os.path.join(experiment_name, run_title) for run_title in run_titles
    ]

    score_arrays = []
    good_run_titles = []
    for log_dir, run_title in zip(log_dirs, run_titles):
        try:
            scores = get_scores(log_dir,
                    subdir=subdir,
                    only_longest=only_longest,
                    min_length=min_length, cumulative=cumulative)

            if len(scores) < min_seeds:
                raise Exception(f"Skipping {run_title} because only {len(scores)} seeds")
            
            if max_episodes > 0:
                scores = scores[:,0:max_episodes]

            score_arrays.append(scores)
            good_run_titles.append(run_title)

            if all_seeds:
                for i, score in enumerate(scores):
                    score_arrays.append(np.array([score]))
                    good_run_titles.append(run_title + f"_{i+1}")
        except Exception as e:
            print(f"skipping {log_dir} due to error {e}")
            pass

    if separate_titles:
        title_names = [rt.replace("_", " ") for rt in good_run_titles]
    else:
        title_names = good_run_titles

    if use_colormap:
        [
            generate_plot(score_array, run_title, smoothen=smoothen, color=COLOR_MAP[run_title], linewidth=linewidth)
            for score_array, run_title in zip(score_arrays, title_names)
        ]
    else:
        [
            generate_plot(score_array, run_title, smoothen=smoothen)
            for score_array, run_title in zip(score_arrays, title_names)
        ]

    if big:
        plt.ylabel(subdir.replace("_", " ").title(), size=45)
        plt.xlabel("Episode", size=45)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        # plt.legend()
        # plt.legend(loc="center right", prop={"size": 40})
        if include_legend:
            plt.legend(loc=legend_loc, prop={"size": 40})
        # plt.legend(loc="lower right", prop={"size": 40})
        # plt.legend(loc="upper right", prop={"size": 40})
    else:
        plt.ylabel(subdir.replace("_", " ").title())
        plt.xlabel("Episode")
        # plt.xticks(fontsize=30)
        # plt.yticks(fontsize=30)
        if include_legend:
            plt.legend(fontsize=8, loc=legend_loc)
        # plt.legend(loc="lower right")
    # plt.tight_layout()
    # plt.xlim((-8., 150.))
    # plt.xlim((-8., 300.))
    # plt.ylim((-0.00001, 0.00002))
    if title:
        plt.title(title, size=40)
    # plt.title("MountainCar", size=40)
    # plt.title(Path(experiment_name).name.replace("_", " ").title())
    if show:
        plt.show()

def all_run_titles(experiment_name):
    parent = Path(experiment_name)
    run_titles = [d.name for d in parent.iterdir()]
    print(run_titles)
    return run_titles

def all_runs_and_subruns(experiment_name):
    parent = Path(experiment_name)
    run_titles = []
    for d in parent.iterdir():
        if d.is_dir():
            run_titles.append(d.name)
            for sub_d in d.iterdir():
                if sub_d.is_dir():
                    run_titles.append(f"{d.name}/{sub_d.name}")
    return run_titles


def make_collated_plots():
    experiment_name = "./local_plots/dm_control/collated_2"
    for i, environment in enumerate(["pendulum", "cartpole", "acrobot", "ball_in_cup"]):
        plt.subplot(2,2, i+1)
        this_exp = f"{experiment_name}/{environment}"
        run_titles = all_run_titles(this_exp)
        make_graphs(this_exp,
            'scores',
            run_titles=run_titles,
            smoothen=True,
            min_length=-1,
            only_longest=False,
            cumulative=False,
            show=False)
    
    plt.show()
    exit()

def make_acrobot_plot():
    experiment_name = "./collated_plots/dm_control/acrobot/"
    run_titles = ["DOIE_(Ours)", "RND", "MPE", "OptBias", "RBFDQN"]
    run_titles = list(reversed(run_titles))
    # run_titles = all_run_titles(experiment_name)
    make_graphs(experiment_name,
                "scores",
                run_titles=run_titles,
                smoothen=25,
                max_episodes=1500,
                big=True,
                separate_titles=True,
                title="Acrobot",
                use_colormap=True,
                include_legend=True,
                legend_loc="upper left",
                linewidth=16)
    exit()

def make_pendulum_plot():
    experiment_name = "./collated_plots/dm_control/pendulum/"
    run_titles = ["DOIE_(Ours)", "RND", "MPE", "OptBias", "RBFDQN"]
    run_titles = list(reversed(run_titles))
    # run_titles = all_run_titles(experiment_name)
    make_graphs(experiment_name,
                "scores",
                run_titles=run_titles,
                smoothen=20,
                max_episodes=500,
                big=True,
                separate_titles=True,
                title="Pendulum",
                use_colormap=True,
                include_legend=False,
                linewidth=16)
    exit()

def make_ball_in_cup_plot():
    experiment_name = "./collated_plots/dm_control/ball_in_cup/"
    run_titles = ["DOIE_(Ours)", "RND", "MPE", "OptBias", "RBFDQN"]
    run_titles = list(reversed(run_titles))
    # run_titles = all_run_titles(experiment_name)
    make_graphs(experiment_name,
                "scores",
                run_titles=run_titles,
                smoothen=20,
                max_episodes=300,
                big=True,
                separate_titles=True,
                title="Ball In Cup",
                use_colormap=True,
                include_legend=False,
                linewidth=16)
    exit()

def make_cartpole_plot():
    experiment_name = "./collated_plots/dm_control/cartpole/"
    run_titles = ["DOIE_(Ours)", "RND", "MPE", "OptBias", "RBFDQN"]
    run_titles = list(reversed(run_titles))
    # run_titles = all_run_titles(experiment_name)
    make_graphs(experiment_name,
                "scores",
                run_titles=run_titles,
                smoothen=30,
                max_episodes=1000,
                big=True,
                separate_titles=True,
                title="Cartpole",
                use_colormap=True,
                include_legend=False,
                linewidth=16)
    exit()

def make_mcar_plot():
    experiment_name = "./collated_plots/cmcar/shaping_runs/"
    run_titles = ["Q-Shaping", "Uniform_Optimism", "Anti_Q-Shaping", "Reward_Shaping", "OptBias"]
    # run_titles = list(reversed(run_titles))
    # run_titles = all_run_titles(experiment_name)
    make_graphs(experiment_name,
                "scores",
                run_titles=run_titles,
                smoothen=15,
                max_episodes=160,
                big=True,
                separate_titles=True,
                title="MountainCar",
                use_colormap=True,
                include_legend=True,
                legend_loc="center right",
                linewidth=16)
    exit()

def make_point_maze_score_plot():
    experiment_name = "./collated_plots/point_maze/full_runs/"
    run_titles = ["DOIE_(Ours)", "OMEGA", "RND", "MPE", "OptBias", "RBFDQN", "Random_Agent"]
    run_titles = list(reversed(run_titles))
    # run_titles = all_run_titles(experiment_name)
    make_graphs(experiment_name,
                "scores",
                run_titles=run_titles,
                smoothen=50,
                max_episodes=2000,
                big=True,
                separate_titles=True,
                title="Point Maze",
                use_colormap=True,
                include_legend=True,
                legend_loc="upper left",
                linewidth=16)
    exit()

def make_point_maze_grid_exploration_amounts_plot():
    experiment_name = "./collated_plots/point_maze/full_runs/"
    run_titles = ["DOIE_(Ours)", "OMEGA", "RND", "MPE", "OptBias", "RBFDQN", "Random_Agent"]
    run_titles = list(reversed(run_titles))
    # run_titles = all_run_titles(experiment_name)
    make_graphs(experiment_name,
                "grid_exploration_amounts",
                run_titles=run_titles,
                smoothen=50,
                max_episodes=2000,
                big=True,
                separate_titles=True,
                title="Point Maze",
                use_colormap=True,
                include_legend=True,
                legend_loc="lower right",
                linewidth=16)
    exit()

def make_exploration_tradeoff_plot(num_eps=-1):
    experiment_name = "./local_plots/point_maze/filter_sweep_2"
    # experiment_name = "./remote_plots/slurm_plots/point_maze/filter_sweep"
    # experiment_name = "./remote_plots/slurm_plots/point_maze/filter_sweep_more"

    run_titles = all_run_titles(experiment_name)

    log_dirs = [
        os.path.join(experiment_name, run_title) for run_title in run_titles
    ]

    exploration_amounts = []
    filter_amounts = []
    for log_dir, run_title in zip(log_dirs, run_titles):
        expl_amount = get_scores(log_dir,
                    subdir="grid_exploration_amounts",
                    only_longest=False,
                    min_length=-1, cumulative=False)
        # __import__('ipdb').set_trace()
        # print('bing')
        # final_expl_amount = expl_amount[:,-1].mean()
        filter_amount = get_scores(log_dir,
                    subdir="filter_percentage",
                    only_longest=False,
                    min_length=-1, cumulative=False)

        if num_eps > 0:
            expl_amount = expl_amount[:,0:num_eps]
            filter_amount = filter_amount[:,0:num_eps]

        # final_filter_amount = filter_amount[:,-1].mean()


        for _ in expl_amount[:,-1]:
            exploration_amounts.append(_)
        for _ in filter_amount[:,-1]:
            filter_amounts.append(_)

        # exploration_amounts.append(final_expl_amount)
        # filter_amounts.append(final_filter_amount)

        # assert expl_amount.shape[1] == 500
        # assert filter_amount.shape[1] == 500

    filter_amounts = [1 - e for e in filter_amounts]

    plt.plot(np.unique(exploration_amounts), np.poly1d(np.polyfit(exploration_amounts, filter_amounts, 1))(np.unique(exploration_amounts)),
             lw=16)
    plt.scatter(exploration_amounts, filter_amounts, c="g", linewidths=10)

    plt.xlabel("Exploration Amounts at 1,000 episodes", size=40)
    plt.ylabel("Fraction Filtered", size=40)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.title("Point Maze", size=40)

    plt.show()

    # print('bong')
    # __import__('ipdb').set_trace()
    # print('bing')


    # print('o[d')
    exit()

def make_value_viz_plots(experiment_name, run_title, use_novelty, seed, episode, point_maze=True):
    """Annoyingly, recreate some of the stuff.
    Or, maybe just add it to the XYPlotter!"""
    full_experiment_name = os.path.join(experiment_name, run_title)
    pkl_string = "seed_{}_episode_{}_q_{}_novelty.pkl".format(seed, episode, "with" if use_novelty else "without")
    load_path = os.path.join(full_experiment_name, "value_plots", pkl_string)

    if point_maze:
        from rbfdqn.utils import make_env
        state_plotter = XYStatePlotter(env=make_env("SibRivPointMaze-v1"))
    else:
        state_plotter = XYStatePlotter(env=None)
    state_plotter.make_q_value_plot_from_log(load_path=load_path, save=False, point_maze=point_maze)

def make_quiver_plots(experiment_name, run_title, seed, episode):
    full_experiment_name = os.path.join(experiment_name, run_title)
    pkl_string = "seed_{}_episode_{}_quiver_plot.pkl".format(seed, episode)
    load_path = os.path.join(full_experiment_name, "value_plots", pkl_string)

    state_plotter = XYStatePlotter(env=None)
    state_plotter.make_quiver_plot_from_log(load_path=load_path, save=False)


# def make_final_mcar_plot():


def main():
    # make_collated_plots()
    # make_acrobot_plot()
    # make_pendulum_plot()
    # make_ball_in_cup_plot()
    # make_cartpole_plot()
    # make_mcar_plot()
    # make_point_maze_score_plot()
    # make_exploration_tradeoff_plot()
    # make_point_maze_grid_exploration_amounts_plot()
    """
    Change these options and directories to suit your needs
    """
    ## Defaults
    subdir = "scores"
    smoothen = False
    min_length = -1
    only_longest = False
    cumulative = False
    all_seeds = False
    min_seeds = 1
    max_episodes = -1

    # Options
    # subdir = "episodic_rewards"
    # subdir = "exploration_amounts"
    # subdir = "training_times"
    # subdir = "all_times"
    # subdir = "average_q_buffer"
    # subdir = "average_q_target_buffer"
    # subdir = "average_loss"
    # subdir = "grid_exploration_amounts"
    # subdir = "filter_percentage"
    # smoothen = True
    # min_length = 21
    # only_longest = True
    # cumulative = True
    # all_seeds = True
    # min_seeds = 4
    # max_episodes = 160


    experiment_name = "./plots/example_experiment/"
    run_titles = ["lr_01", "lr_05", "lr_10"]
    make_graphs(experiment_name,
                subdir,
                run_titles=run_titles,
                smoothen=smoothen,
                min_length=min_length,
                only_longest=only_longest,
                cumulative=cumulative,
                all_seeds=all_seeds,
                min_seeds=min_seeds,
                max_episodes=max_episodes,
                big=True,
                separate_titles=True)

if __name__ == "__main__":
    main()

