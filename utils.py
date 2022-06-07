import pandas as pd
import numpy as np
import datetime as dt
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from time import time, sleep
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

class Account:
    def __init__(self, buy_amount=1000.0, fee=0.1/100, asset='BTC', qty_dec_places=8,
                 quote_asset='USD', price_dec_places=2, verbose=False):
        self.buy_amount = buy_amount
        self.fee = fee
        self.asset = asset
        self.quote_asset = quote_asset
        self.verbose = verbose
        self.owned_qty = 0.0
        self.overall_profits = 0.0
        self.no_of_trades = 0
        self.recorded_profits = []
        self.current_value = []
        self.position_open = False
        self.buy_triggered = False
        self.sell_triggered = False
        self.buy_message = '{:s} bought {:.' + str(qty_dec_places) \
                           + 'f} {:s} at {:.' + str(price_dec_places) + 'f} {:s}'
        self.sell_message = '{:s} sold all at {:.' + str(price_dec_places) \
                            + 'f} {q_ass:s}, profit: {:.' + str(price_dec_places) \
                            + 'f} {q_ass:s} (overall thus far: {:.' \
                            + str(price_dec_places) + 'f} {q_ass:s})\n'
    
    def buy(self, quantity, price, timestamp):
        if not self.position_open:
            self.no_of_trades += 1
            self.position_open = True
            self.buy_triggered = False
            self.owned_qty = quantity * (1.0 - self.fee)
            self.last_buy_price = price
            self.last_buy_value = quantity * price
            if self.verbose:
                print(self.buy_message.format(timestamp, self.owned_qty, self.asset, price, self.quote_asset))
        else:
            if self.verbose:
                print('buy order not executed; position already open')
    
    def sell_all(self, price, timestamp):
        if self.position_open:
            self.no_of_trades += 1
            self.position_open = False
            self.sell_triggered = False
            sell_value = self.owned_qty * price
            profit = sell_value * (1.0 - self.fee) - self.last_buy_value
            self.overall_profits += profit
            self.recorded_profits.append(profit)
            self.owned_qty = 0.0
            if self.verbose:
                print(self.sell_message.format(timestamp, price, profit, self.overall_profits, q_ass=self.quote_asset))
        else:
            if self.verbose:
                print('sell order not executed; position already closed')

def hodl_profits(df, buy_amount=1000.0):
    return buy_amount * (df.iloc[-1]['close'] / df.iloc[0]['open'] - 1.0)

def backtest_mlmodel(df, dfml, y_pred, buy_amount=1000.0, investment_style='fixed', reinvest_rate=1.0,
                     buy_thresh=0.05, sell_thresh=-0.05, stoploss_enabled=True, sl_pct_thresh=15.0, sl_atr_factor=2.0,
                     sl_timeout_enabled=True, sl_timeout_hours=2, buy_scalping_enabled=True,
                     asset='BTC', quote_asset='USD', qty_dec_places=8, price_dec_places=2,
                     verbose=0, chart_enabled=False, chart_logscale=True):
    
    reinvest_rate = abs(reinvest_rate)
    if reinvest_rate > 1.0: raise ValueError("reinvest_rate can't be higher than 1")
    
    acc = Account(
        buy_amount = buy_amount,
        asset = asset,
        quote_asset = quote_asset,
        qty_dec_places = qty_dec_places,
        price_dec_places = price_dec_places,
        verbose = verbose >= 2
    )
    buy_amount_actual = buy_amount
    
    reopen_time = dt.datetime(1, 1, 1, tzinfo=dt.timezone.utc)
    
    for index, row in dfml.iterrows():
        if acc.position_open:
            current_value = buy_amount + acc.overall_profits \
                            + acc.owned_qty * df.loc[index, 'open'] * (1.0 - acc.fee) - acc.last_buy_value
        else:
            current_value = buy_amount + acc.overall_profits
        acc.current_value.append(current_value)
        
        if buy_amount_actual <= 0.0:
            if verbose >= 2: print("Congratulations! You're broke.")
            break
        
        if stoploss_enabled and sl_timeout_enabled and index < reopen_time:
            continue
        
        if acc.buy_triggered:
            if df.loc[index, 'low'] <= buy_target_price:
                acc.buy(buy_amount_actual/buy_target_price, buy_target_price, str(index))
                if stoploss_enabled:
                    stoploss_level = (df.loc[index, 'ema10'] - sl_atr_factor * df.loc[index, 'atr10']) \
                                * (1. - sl_pct_thresh / 100.)
            else:
                buy_scalp_width /= 2.0
                buy_target_price = buy_signal_price - buy_scalp_width
        
        elif acc.sell_triggered:
            acc.sell_all(df.loc[index, 'open'], str(index))
            if investment_style == 'cumulative':
                buy_amount_actual = buy_amount + acc.overall_profits * min(reinvest_rate, 1.0)
            elif investment_style == 'floor':
                buy_amount_actual = buy_amount + max(acc.overall_profits * reinvest_rate, 0.0)
        
        if acc.position_open and stoploss_enabled:
            if df.loc[index, 'low'] <= stoploss_level:
                if verbose >= 2:
                    print('{:s} stop-loss triggered!'.format(str(index)))
                sell_price = stoploss_level if df.loc[index, 'open'] >= stoploss_level else df.loc[index, 'open']
                acc.sell_all(sell_price, str(index))
                if investment_style == 'cumulative':
                    buy_amount_actual = min(
                        buy_amount + reinvest_rate * acc.overall_profits,
                        buy_amount + acc.overall_profits
                    )
                elif investment_style == 'floor':
                    buy_amount_actual = max(acc.buy_amount, acc.buy_amount + acc.overall_profits * reinvest_rate)
                if sl_timeout_enabled:
                    reopen_time = index + dt.timedelta(hours=sl_timeout_hours)
                continue
            
            sl_new = (df.loc[index, 'ema10'] - sl_atr_factor * df.loc[index, 'atr10']) * (1. - sl_pct_thresh / 100.)
            if sl_new > stoploss_level:
                stoploss_level = sl_new
        
        if acc.position_open or acc.buy_triggered:
            if row[y_pred] < sell_thresh:
                if acc.buy_triggered:
                    acc.buy_triggered = False
                    if verbose >= 2:
                        print('{:s} buy order canceled\n'.format(str(index)))
                else:
                    acc.sell_triggered = True
                    if verbose >= 2:
                        print('{:s} triggering sell order'.format(str(index)))
        elif row[y_pred] > buy_thresh:
            acc.buy_triggered = True
            buy_signal_price = df.loc[index, 'close']
            if buy_scalping_enabled:
                buy_scalp_width = 40 + 0.0005 * df.loc[index, 'atr10'] ** 2
            else:
                buy_scalp_width = 0.0
            buy_target_price = buy_signal_price - buy_scalp_width
            if verbose >= 2:
                print('{:s} triggering buy order'.format(str(index)))
    
    if verbose >= 2: print()
    if verbose >= 1:
        if acc.position_open:
            unrealized_pnl = acc.owned_qty * df.iloc[-1]['close'] * (1.0 - acc.fee) - acc.last_buy_value
        else:
            unrealized_pnl = 0.0
        profits_array = np.array(acc.recorded_profits)
        cumulative_profits = np.cumsum(profits_array)
        win_index = np.where(profits_array >= 0)
        loss_index = np.where(profits_array < 0)
        avg_win = profits_array[win_index].mean()
        avg_loss = profits_array[loss_index].mean()
        
        print((
            'overall profits: {:.{pr_dec:d}f} {q_ass:s} ' \
            + '({:.{pr_dec:d}f} {q_ass:s} realized + {:.{pr_dec:d}f} {q_ass:s} unrealized)\n' \
            + ' '*7 + 'vs. HODL: {:.{pr_dec:d}f} {q_ass:s}'
            ).format(
                acc.overall_profits + unrealized_pnl,
                acc.overall_profits,
                unrealized_pnl,
                hodl_profits(df, buy_amount),
                pr_dec = price_dec_places,
                q_ass = quote_asset
            )
        )
        print(
            '\n best point: {:+.{pr_dec:d}f} {q_ass:s}\nworst point: {:+.{pr_dec:d}f} {q_ass:s}'.format(
                cumulative_profits.max(),
                cumulative_profits.min(),
                pr_dec = price_dec_places,
                q_ass = quote_asset
            )
        )
        print('\n# of trades (overall): {:d}'.format(acc.no_of_trades))
        print(
            '        winning ratio: {:.1f}%'.format(
                100. * win_index[0].shape[0] / (win_index[0].shape[0] + loss_index[0].shape[0])
            )
        )
        print('\n  average win: {:+.{:d}f} {:s}'.format(avg_win, price_dec_places, quote_asset))
        print(' average loss: {:+.{:d}f} {:s}'.format(avg_loss, price_dec_places, quote_asset))
        print('        ratio:  {:.2f}'.format(avg_win / abs(avg_loss)))
        
        if chart_enabled:
            acc_performance = np.array(acc.current_value) / buy_amount * df.iloc[0]['open']
            acc_performance = pd.Series(acc_performance, index=df.index)
            fig = plt.figure(figsize=(15, 8))
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
            ax0 = fig.add_subplot(gs[0])
            ax0.plot(df.loc[:, 'close'])
            ax0.plot(acc_performance, c='orange')
            ax0.set_xlim([df.index[0], df.index[-1]])
            ax0.legend(['asset value', 'account performance'])
            plt.setp(ax0.get_xticklabels(), visible=False)
            if chart_logscale:
                ax0.set_yscale('log')
            ax1 = fig.add_subplot(gs[1], sharex=ax0)
            ax1.plot(dfml.index, np.full(dfml.shape[0], buy_thresh), c='green', lw=1.0)
            ax1.plot(dfml.index, np.full(dfml.shape[0], 0.0), c='black', lw=0.5)
            ax1.plot(dfml.index, np.full(dfml.shape[0], sell_thresh), c='red', lw=1.0)
            ax1.plot(dfml.loc[:, y_pred], c='#004080', lw=1.)
            ax1.set_ylim(sell_thresh - 0.06, buy_thresh + 0.06)
            plt.show()
    
    return acc

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
