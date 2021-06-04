from gym.envs.registration import register



register(
    id='SibRivPointMaze-v1',  #120.hyper
    entry_point='rbfdqn.tasks.sibrivmaze.maze_env:Env',
    max_episode_steps=
    50,  # From SibRiv paper... makes it hard cause best policy is like 20 I think.
    reward_threshold=-3.75,  # This just doesn't matter.
    kwargs={
        'n':
        1000,  # This is how they do time-limit-truncation, but I let Gym handle that.
        'fixed_goal': (9., 9.)
    }
    #         'n': 10,

    #         'maze_id': 'd4rl-maze',
    #         'n_bins': 0,
    #         'observe_blocks': False,
    #         'put_spin_near_agent': False,
    #         'top_down_view': False,
    #         'manual_collision': True,
    #         'maze_size_scaling': 3,
    #         'color_str': "0 1 0 1",
    #     }
)

register(
    id="PlaneWorld-v1",
    entry_point="rbfdqn.tasks.simple_nav:PlaneWorldEnv",
    max_episode_steps=50,
    reward_threshold=10.,
)
