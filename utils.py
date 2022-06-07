import pandas as pd
import numpy as np
import datetime as dt
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from time import time, sleep
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

def gridsearch_xgb(X, y, param_grid, base_params, save_path=None):
    n_computations = 5
    for list in param_grid.values():
        n_computations *= len(list)

    n_hours = n_computations / (700.0 * 600.0 / np.mean(param_grid['n_estimators']))
    n_minutes = (n_hours % 1) * 60.0
    if int(n_hours) > 0:
        hour_string = '{:d} hour{:s} and '.format(int(n_hours), '' if int(n_hours) == 1 else 's')
    else:
        hour_string = ''
    minute_string = '{:d} minute{:s}'.format(int(round(n_minutes)), '' if round(n_minutes) == 1 else 's')
    reply = input('Grid search with these parameters will take {:d} fits, or approximately {:s}{:s}. Is that ok? (y/N) '.format(n_computations, hour_string, minute_string))

    if reply == 'y':
        start_time = time()
        grid = GridSearchCV(
            XGBRegressor(**base_params),
            param_grid = param_grid,
            scoring = ['neg_root_mean_squared_error', 'r2', 'neg_median_absolute_error'],
            cv = 5,
            refit = 'neg_root_mean_squared_error',
            verbose = 3
        ).fit(X, y)
        end_time = time()
        print('true elapsed time: {:.1f}min'.format((end_time - start_time) / 60))
        if save_path:
            joblib.dump(grid, save_path)
        rmse_max = grid.cv_results_['mean_test_neg_root_mean_squared_error'].max()
        index_best_rmse = [i for i in range(len(grid.cv_results_['mean_test_neg_root_mean_squared_error'])) if grid.cv_results_['mean_test_neg_root_mean_squared_error'][i] == rmse_max]
        print('\nbest RMSE:', rmse_max)
        for index in index_best_rmse: print(grid.cv_results_['params'][index])
        r2_max = grid.cv_results_['mean_test_r2'].max()
        index_best_r2 = [i for i in range(len(grid.cv_results_['mean_test_r2'])) if grid.cv_results_['mean_test_r2'][i] == r2_max]
        print('\nbest R2:', r2_max)
        for index in index_best_r2: print(grid.cv_results_['params'][index])
        mad_max = grid.cv_results_['mean_test_neg_median_absolute_error'].max()
        index_best_mad = [i for i in range(len(grid.cv_results_['mean_test_neg_median_absolute_error'])) if grid.cv_results_['mean_test_neg_median_absolute_error'][i] == mad_max]
        print('\nbest MAD:', mad_max)
        for index in index_best_mad: print(grid.cv_results_['params'][index])
        return grid
    else:
        print('Canceled')

def print_top_n(grid, n=10, metric='rmse'):
    metric_dict = {
        'rmse_mean': 'mean_test_neg_root_mean_squared_error',
        'rmse_std': 'std_test_neg_root_mean_squared_error',
        'r2_mean': 'mean_test_r2',
        'r2_std': 'std_test_r2',
        'mad_mean': 'mean_test_neg_median_absolute_error',
        'mad_std': 'std_test_neg_median_absolute_error'
    }
    metric_mean = metric + '_mean'
    metric_std = metric + '_std'
    
    score_df = pd.DataFrame(
        [
            grid.cv_results_['params'],
            grid.cv_results_[metric_dict[metric_mean]],
            grid.cv_results_[metric_dict[metric_std]]
        ]
    ).T
    score_df.columns = ['params', metric_mean, metric_std]
    
    for index, row in score_df.iloc[np.argsort(score_df[metric_mean])][::-1].iloc[:n].iterrows():
        print('{:s}: {:.6f} ± {:.6f}'.format(metric, row[metric_mean], row[metric_std]))
        print(row['params'])
        print()

def plot_scores_1d(grid, smoothing_window=5, xtick_step=1):
    metrics = {
        'mean_test_neg_root_mean_squared_error': 'root mean squared error',
        'mean_test_r2': 'R2 score',
        'mean_test_neg_median_absolute_error': 'median absolute deviation'
    }
    
    n_varied_params = 0
    for param_name, values in grid.param_grid.items():
        if len(values) > 1:
            param = param_name
            n_varied_params += 1
    
    if n_varied_params > 1:
        raise ValueError('This function only supports GridSearchCV results with a single varied parameter')
    
    fig, axes = plt.subplots(len(metrics), 2, figsize=(15, 15))
    for ax_pair, metric in zip(axes, metrics.keys()):
        for ax, plot_type in zip(ax_pair, ['abs', 'dif']):
            if plot_type == 'abs':
                ax.plot(grid.param_grid[param], grid.cv_results_[metric])
                ax.set_xticks(grid.param_grid[param][::xtick_step])
                ax.set_xlim(np.min(grid.param_grid[param]), np.max(grid.param_grid[param]))
                ax.title.set_text(metrics[metric])
            else:
                dif = pd.Series(
                    np.diff(grid.cv_results_[metric], 1),
                    index = grid.cv_results_['param_' + param][1:]
                ).rolling(smoothing_window).mean()
                ax.plot(dif)
                if dif.min() < 0.0 and dif.max() > 0.0:
                    ax.plot(dif.loc[dif.notna()].index, np.zeros(dif.loc[dif.notna()].shape[0]), c='red', lw=1.0)
                ax.set_xticks(grid.param_grid[param][::xtick_step])
                ax.set_xlim(dif.loc[dif.notna()].index.min(), dif.loc[dif.notna()].index.max())
                ax.title.set_text(metrics[metric] + ' differential (smoothed across {:d})'.format(smoothing_window))
    fig.suptitle('Scores depending on {:s}'.format(param), fontsize=14, y=0.928)
    plt.show()
    return fig

def plot_scores_2d(grid, param_x=None, param_y=None):
    metrics = {
        'mean_test_neg_root_mean_squared_error': 'root mean squared error',
        'mean_test_r2': 'R2 score',
        'mean_test_neg_median_absolute_error': 'median absolute deviation'
    }
    
    if param_x is None or param_y is None:
        varied_params = []
        for param_name, values in grid.param_grid.items():
            if len(values) > 1: varied_params.append(param_name)
        
        param_x = varied_params[0] if len(grid.param_grid[varied_params[0]]) < len(grid.param_grid[varied_params[1]]) else varied_params[1]
        param_y = varied_params[1] if param_x != varied_params[1] else varied_params[0]
        
        if len(varied_params) != 2:
            raise ValueError('This GridSearchCV result has more than two varied parameters; please specify the two to use for this plot')
    
    x = grid.cv_results_['param_' + param_x]
    y = grid.cv_results_['param_' + param_y]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (ax, metric) in enumerate(zip(axes, metrics.keys())):
        z = grid.cv_results_[metric]
        heatmap_data = pd.DataFrame({'x': x, 'y': y, 'z': z}).pivot('y', 'x', 'z')
        sns.heatmap(heatmap_data, ax=ax, cmap=plt.cm.coolwarm)
        ax.title.set_text(metrics[metric])
        ax.set_xlabel(param_x)
        if i < 1: ax.set_ylabel(param_y)
        else: ax.set_ylabel('')
    plt.show()
    return fig
