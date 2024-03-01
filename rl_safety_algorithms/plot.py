import os
import argparse
import warnings

import numpy as np
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import json
from rl_safety_algorithms.common.experiment_analysis import get_file_contents, \
    get_experiment_paths, ParameterContainer, ExperimentAnalyzer
from cycler import cycler


# === Plot settings
matplotlib.rcParams['text.usetex'] = True


def convert_dict_to_tex_label(setup: dict) -> str:
    """Use LaTeX symbols instead of string keys."""
    if isinstance(setup, str):
        return setup  # no need to convert strings...
    key2symbol = {
        'lam': r'\lambda',
        'lambda_lr': r'\alpha_\lambda',
        'target_kl': r'\delta',
        'lam_c': r'\lambda_c',
        'pi_lr': r'\alpha_\pi'
    }
    label_string = r''
    for k, v in setup.items():
        if k in key2symbol:
            label_string += fr'${key2symbol[k]} = {str(v)}$ '
        else:
            label_string += fr'${k} = {str(v)}$ '
    return label_string


def smooth(x, window_len=21, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', "
                         "'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    y = y[(window_len // 2):-(window_len // 2)]
    assert y.shape == x.shape, f'y={y.shape}, x={x.shape}'
    return y



class Plotter:
    def __init__(self,
                 base_dir: str,
                 add_cost_plot: bool = False,
                 alg_setups: dict = {},
                 cost_threshold: float = 25.,
                 plot_best: bool = False,
                 extra_conditions: str = '',
                 window_length: int = 21,
                 select: str = '',  # select algorithms: ['trpo', 'ppo']
                 save_fig: bool = False,  # save figure before plotting
                 y_limit: float = 0.,
                 debug: bool = True
    ):
        self.plot_best = plot_best
        # cut-off of "/" in path name
        self.base_dir = base_dir if base_dir[-1] != os.sep else base_dir[:-1]
        self.add_cost_plot = add_cost_plot
        self.env_id = None
        self.cost_threshold = cost_threshold
        self.save_fig = save_fig
        self.window_length = window_length
        self.debug = debug

        self.current_run_dir = None
        self.cost_analyzer = None
        self.ret_analyzer = None
        self.progress_analyzer = None
        self.y_limit = y_limit
        self.unconstrained_algs = ['trpo', 'ppo', 'npg', 'iwpg']

        # automatically detect information if not provided
        self.alg_setups = self.detect_algorithm_setup() \
            if not alg_setups else alg_setups

        print(self.alg_setups) if debug else None

        # visualization settings
        self.colors = sns.color_palette()

    def detect_algorithm_setup(self) -> dict:
        """ Automatically detect algorithmic setups from configuration files.

        Walks recursively through the provided base directory and opens
        config.json files.

        Returns
        -------
        {'cpo': {'lam_c': [0.9, 0.5, 0.95], 'target_kl': [0.01, 0.005]},
         'trpo': {'target_kl': [0.02, 0.01, 0.005]}
        }
        """
        i = 0
        unified_alg_setup = {}
        experiment_paths = get_experiment_paths(self.base_dir)
        for path in experiment_paths:
            config_file_path = os.path.join(path, 'config.json')
            config = get_file_contents(config_file_path)
            algorithm_name = config.pop('alg')
            # algorithm_name = 'iwpg'  # todo: delete line
            # discard irrelevant information from configuration
            config.pop('logger_kwargs')
            config.pop('exp_name')
            config.pop('seed')
            if self.env_id is None:
                self.env_id = config.pop('env_id')
            else:
                assert self.env_id == config.pop('env_id'), \
                    'Multiple environments in path.'

            def make_hashable(val):
                if isinstance(val, dict):
                    # dictionaries are not hashable, so convert to strings
                    v = str(val)
                elif isinstance(val, list) and len(val) == 1:
                    # lists are also un-hashable types
                    v = val[0]
                else:
                    v = val
                return v

            if algorithm_name not in unified_alg_setup:
                # first time that algorithm is found in config files
                value_set_dict = dict()
                for key, val in config.items():
                    try:
                        v = make_hashable(val)
                        value_set_dict[key] = set([v, ])
                    except:
                        warnings.warn(f"cannot hash: {key}:\n{val}")
                unified_alg_setup[algorithm_name] = value_set_dict

            else:
                # algorihm in detected_alg_setup
                # now add new configs if found
                d = unified_alg_setup[algorithm_name]
                for key, val in config.items():
                    try:
                        v = make_hashable(val)
                        d[key].add(v)
                    except:
                        warnings.warn(f"cannot hash: {key}:\n{val}")
            i += 1

        # detect all config values that have at least 2 different values
        detected_alg_setup = {}
        self.print(f'===== Detected Algorithm Setups =====')
        for alg, dic in unified_alg_setup.items():
            self.print(f'{alg.upper()}:')
            d = detected_alg_setup[alg] = {}
            for param, values in dic.items():
                if len(values) > 1:
                    d[param] = list(values)
                    self.print(f'\t- {param}: {d[param]}')
        self.print(f'=====================================')
        return detected_alg_setup

    def determine_best_alg_setups(self) -> dict:

        best_alg_configs = dict()

        for alg, setup in self.alg_setups.items():
            best_ret_score = -np.inf
            best_cost_score = np.inf
            best_config = None
            fulfills_cost_criterion = False
            self.set_run_dir(os.path.join(self.base_dir, alg))

            parameters = tuple(setup.keys())
            for param_values in product(*setup.values()):
                _filter = dict(zip(parameters, param_values))
                self.print(_filter)
                ret = self.ret_analyzer.get_data(parameters, filter=_filter)
                ret_values = [v for v in ret.values()]
                self.print(f'Ret = {np.mean(ret_values)}.')

                if self.add_cost_plot and alg not in self.unconstrained_algs:
                    cost = self.cost_analyzer.get_data(parameters, filter=_filter)
                    cost_values = [v for v in cost.values()]
                    self.print(f'Cost = {np.mean(cost_values)}.')

                    if np.mean(cost_values) <= self.cost_threshold:
                        score = np.mean(ret_values) - np.std(ret_values)
                        if score > best_ret_score:
                            best_ret_score = score
                            best_cost_score = np.mean(cost_values)
                            best_config = _filter
                            fulfills_cost_criterion = True
                            self.print(f'fulfills_cost_criterion: {fulfills_cost_criterion}')
                    else:
                        if np.mean(cost_values) < best_cost_score \
                                and not fulfills_cost_criterion:
                            best_cost_score = np.mean(cost_values)
                            best_config = _filter
                else:  # no cost plot required or un-constrained algorithm
                    score = np.mean(ret_values) - np.std(ret_values)
                    if score > best_ret_score:
                        best_ret_score = score
                        best_config = _filter

            # best save best config for each algorithm
            best_config['fulfills_criterion'] = fulfills_cost_criterion
            best_alg_configs[alg] = best_config
            self.print("best_alg_configs:")
            self.print(best_alg_configs)
        return best_alg_configs

    def plot(self, y: str = 'EpRet/Mean'):
        if self.plot_best:
            # plot only the best hyper-parameter setup averaged over multiple
            # seeds
            best_alg_setups = self.determine_best_alg_setups()
            self.plot_alg_setup(best_alg_setups)
        elif self.add_cost_plot:
            self.plot_single_algorithm_with_costs()

        else:
            self.plot_value_of_interest(y)

    def print(self, msg):
        if self.debug:
            print(msg)

    def set_run_dir(self, new_run_dir):
        """ Updates experiment analyzers if new run directory is provided.
        """
        if self.progress_analyzer is None or \
                not self.progress_analyzer.base_dir == new_run_dir:
            self.progress_analyzer = ExperimentAnalyzer(
                base_dir=new_run_dir,
                data_file_name='progress.csv',
                debug=self.debug
            )
        if self.ret_analyzer is None or \
                not self.ret_analyzer.base_dir == new_run_dir:
            self.ret_analyzer = ExperimentAnalyzer(
                base_dir=new_run_dir,
                data_file_name='returns.csv',
                debug=self.debug
            )
        if self.add_cost_plot and (self.cost_analyzer is None or not self.cost_analyzer.base_dir == new_run_dir):
            self.cost_analyzer = ExperimentAnalyzer(
                base_dir=new_run_dir,
                data_file_name='costs.csv',
                debug=self.debug
            )

    def get_data(self, query='EpRet/Mean', setup: dict = {}) -> np.ndarray:
        """ fetch data from the experiment analyzers."""
        parameters = tuple(setup.keys())

        if query == 'Eval/Ret':
            ret = self.ret_analyzer.get_data(parameters, filter=setup)
            data = np.array([r for r in ret.values()])
        elif query == 'Eval/Cost':
            cost = self.cost_analyzer.get_data(parameters, filter=setup)
            data = np.array([v for v in cost.values()])
        else:
            try:
                data_dict = self.progress_analyzer.get_data(parameters,
                                                            filter=setup)
                # data holds only one key which is the best configuration.
                # extract values as list
                assert len(data_dict) > 0, f'setup: {setup} got no values'
                pd_list = list(data_dict.values())[0]
                data = np.array([pd[query].values for pd in pd_list])
            except KeyError as e:
                print(f'Key={query} not found in progress.csv')
                raise e

        return data

    def plot_alg_setup(self,
                       alg_setups: dict,
                       display_alg_config=False) -> None:
        """

        Parameters
        ----------
        alg_setups: Dictionary holding algorithm names as keys and parameter configurations
                    as values, e.g.
                    {'rcpo': {'projection': 'KL', 'lam_c': 0.5, 'target_kl': 0.005},
                    'cpo': {'target_kl': 0.01, 'lam_c': 0.95}
                    }
        display_alg_config: shows algorithm hyper-parameters in plot

        Returns
        -------
        None

        """
        N = 2 if self.add_cost_plot else 1
        fig, axes = plt.subplots(1, N, figsize=(4 + 2 * N, 2 * N))
        fig.suptitle(self.env_id)

        for i, (alg, alg_setup) in enumerate(alg_setups.items()):
            fulfills_criterion = alg_setup.pop('fulfills_criterion')
            ls = ':' if self.add_cost_plot and not fulfills_criterion else '-'

            self.set_run_dir(os.path.join(self.base_dir, alg))
            ret = self.get_data(query='EpRet/Mean', setup=alg_setup)

            ax = axes.flatten()[0] if isinstance(axes, np.ndarray) else axes
            self.plot_data(ax=ax, ys=ret, label=alg.upper(), linestyle=ls)
            if self.add_cost_plot:
                ax = axes.flatten()[1]
                costs = self.get_data(query='EpCosts/Mean', setup=alg_setup)
                self.plot_data(ax=ax, ys=costs, label=alg.upper(),
                               linestyle=ls, y_limit=self.y_limit)
        plt.legend()
        plt.show()

    def plot_data(self,
                  ax: plt.Axes,
                  ys: np.ndarray,
                  label: str,
                  title: str = '',
                  # data: pd.DataFrame,
                  xs: None = None,  # takes the shape of ys if not provided
                  x_label: str = 'Epochs',
                  y_label='Performance',
                  window_length: int = 13,
                  linestyle: str = '-',
                  y_limit: float = 0.,
                  add_number_of_seeds_to_legend=True,
                  **kwargs
                  ):
        assert len(ys.shape) == 2 or len(ys.shape) == 1, 'Expected 1D/2D-array.'
        if len(ys.shape) == 2:
            if xs is None:
                xs = np.arange(ys.shape[1])
            y_mean = smooth(np.mean(ys, axis=0), window_len=window_length)
            y_std = smooth(np.std(ys, axis=0), window_len=window_length)
            ax.fill_between(xs, y_mean - y_std, y_mean + y_std,
                            alpha=0.15)
        else:  # len(ys.shape) == 1
            if xs is None:
                xs = np.arange(ys.shape[0])
            y_mean = smooth(ys, window_len=window_length)
        if add_number_of_seeds_to_legend:
            num = ys.shape[0] if len(ys.shape) == 2 else 1
            label += fr' ({num})'
        ax.plot(xs, y_mean, label=label, ls=linestyle, **kwargs)
        # color=self.colors[i])
        if title:
            ax.set_title(title)
        if y_limit > 0:
            ax.set_ylim([0, y_limit])
            ax.plot(xs, self.cost_threshold * np.ones_like(y_mean), 'b--')

    def plot_value_of_interest(self, y):

        # size of plot depends on number of algorithms
        N = int(np.ceil(np.sqrt(len(self.alg_setups.keys()))))
        print(f'N={N}')
        fig, axes = plt.subplots(N, N, figsize=(8 + 2 * N, 8 + 2 * N))
        j = 0

        for alg, alg_setup in self.alg_setups.items():
            ax = axes.flatten()[j] if isinstance(axes, np.ndarray) else axes
            ax.set_title(f'{self.env_id}: {alg.upper()}')
            ax.set_xlabel('Epochs')
            ax.set_ylabel(y)
            # use cyclers to enlarge number of plots greater than 10...
            cc = cycler(linestyle=['-', '--', '-.']) * cycler(color=self.colors)
            ax.set_prop_cycle(cc)

            # check if base dir includes already the algorithm name
            if self.base_dir.split(os.sep)[-1] == alg:
                exp_dir = self.base_dir
            else:
                exp_dir = os.path.join(self.base_dir, alg)

            self.set_run_dir(exp_dir)

            # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            # {'lambda_lr': [0.1, 0.01, 0.05], 'lambda_optimizer': ['SGD', 'Adam']}

            for i, setup_values in enumerate(product(*alg_setup.values())):
                parameters = tuple(alg_setup.keys())
                setup = dict(zip(parameters, setup_values))
                returns = self.get_data(query=y, setup=setup)
                self.plot_data(
                    ax,
                    ys=returns,
                    label=convert_dict_to_tex_label(setup),
                )
            plt.legend()
            j += 1
        file_path = '/var/tmp/figures/'
        os.makedirs(file_path, exist_ok=True)
        file_name_path = os.path.join(file_path, 'figure.pdf')
        plt.savefig(os.path.join(file_path, 'figure.pdf'), bbox_inches='tight')
        print(f'Saved figure to: {file_name_path}')
        plt.show()

    def plot_single_algorithm_with_costs(self):
        assert len(self.alg_setups.keys()) == 1, \
            'Cost plotting is currently only supported for a single algo.'
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        alg = list(self.alg_setups.keys())[0]
        alg_setup = list(self.alg_setups.values())[0]
        if self.base_dir.split(os.sep)[-1] == alg:
            exp_dir = self.base_dir
        else:
            exp_dir = os.path.join(self.base_dir, alg)

        self.set_run_dir(exp_dir)
        fig.suptitle(f'{self.env_id} - {alg}')

        for i, setup_values in enumerate(product(*alg_setup.values())):
            parameters = tuple(alg_setup.keys())
            setup = dict(zip(parameters, setup_values))

            returns = self.get_data(query='EpRet/Mean', setup=setup)
            costs = self.get_data(query='EpCosts/Mean', setup=setup)

            eval_ret = self.get_data('Eval/Ret', setup)
            eval_cost = self.get_data('Eval/Cost', setup)

            print(f'Alg Setup: {setup}')
            print(f'J: \t{np.nanmean(eval_ret):0.1f} '
                  f'H: \t{np.nanmean(eval_cost):0.2f} ')

            self.plot_data(axes[0], ys=returns, label=convert_dict_to_tex_label(setup), title='Returns')
            self.plot_data(axes[1], ys=costs, title='Costs',
                           label=convert_dict_to_tex_label(setup),
                           y_limit=self.y_limit
                           )
        plt.legend()
        plt.show()


def fetch_args():
    n_cpus = os.cpu_count()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--add-cost', action='store_true')
    parser.add_argument('--alg-setup', type=str, default='',
                        help="E.g. \"{'iwpg': {'weight_initialization': ['kaiming_uniform', 'glorot', 'orthogonal']}}\"")
    parser.add_argument('--best', action='store_true')
    parser.add_argument('--cost-threshold', '-c', type=float, default=25.)
    parser.add_argument('--num-runs', '-r', type=int, default=4,
                        help='Number of total runs that are executed.')
    parser.add_argument('--logdir', type=str,
                        help='Define a custom directory for logging.')
    parser.add_argument('--savefig', action='store_true',
                        help='Safe plot as figure.')
    parser.add_argument('--select', '-s', nargs='*',
                        help='Selection rule: the plotter will only show curves'
                             'from logdirs that contain all of these substrings.')
    parser.add_argument('--window-length', '-wl', type=int, default=21,
                        help='Define a custom directory for logging.')
    parser.add_argument('--yaxis', '-y', type=str, default='EpRet/Mean')
    parser.add_argument('--ylim', type=float, default=0.)
    return parser.parse_args()


if __name__ == '__main__':
    args = fetch_args()
    print(args.alg_setup, type(args.alg_setup))
    alg_setups = {} if args.alg_setup == '' else json.loads(args.alg_setup)

    plotter = Plotter(
        base_dir=args.logdir,
        alg_setups=alg_setups,
        plot_best=args.best,
        add_cost_plot=args.add_cost,
        select=args.select,
        cost_threshold=args.cost_threshold,
        window_length=args.window_length,
        save_fig=args.savefig,
        y_limit=args.ylim
    )
    plotter.plot(args.yaxis)
