import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import pywt
import multiprocessing
import time
import os

START_DATE = '2024-06-01'
END_DATE = '2025-04-01'
WINDOW_SIZE = 80
STEP_SIZE = 20

def getAdjCloseData(ticker, start_date, end_date):
    data = pd.read_csv('NumericalData/' + ticker + '.csv', index_col='Date')
    data.index = pd.to_datetime(data.index)
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    return data

def check_for_missing_values(data, start_idx, end_idx, trading_days_skeleton):
    window_length = end_idx - start_idx
    subset = data.iloc[start_idx:end_idx, 5:6]
    subset_start_date = subset.index[0]
    trading_days_skeleton_start_index = trading_days_skeleton[trading_days_skeleton['Date'] == subset_start_date].index[0]
    trading_days_skeleton_subset = trading_days_skeleton.iloc[trading_days_skeleton_start_index:trading_days_skeleton_start_index + window_length]
    trading_days_skeleton_subset.set_index('Date', inplace=True)
    subset = pd.merge(trading_days_skeleton_subset, subset, left_index=True, right_index=True, how='left')
    if subset.shape[0] != window_length:
        return True,
    if subset.isnull().values.any():
        return True
    return False

def generate_time_series(data, start_idx, end_idx, step_idx):
    subset = data.iloc[start_idx:end_idx, 5]
    t = subset.index[-1].strftime('%Y-%m-%d')
    ts = subset.values
    last_ts_value = ts[-1]
    targets = data.iloc[end_idx:step_idx, 5].values[[-20, -16, -11, -6, -1]]
    return t, ts, last_ts_value, targets

def normalize_time_series(ts):
    return (ts - np.min(ts)) / (np.max(ts) - np.min(ts))

def generate_sign_intensities(ts):
    ts = (ts * 255).astype(np.uint8)
    ts = ts.reshape(1, 80)
    return ts

def generate_continuous_wavelet_transform_coefficients(ts, scales):
    wavelet = 'cmor2.0-1.0'
    coefficients, _ = pywt.cwt(ts * np.sqrt(2) * np.pi ** (1/4), scales, wavelet)
    return coefficients

def generate_file_name(t, last_ts_value, targets, ticker):
    return f'VolumeSpectrograms/{ticker}/{ticker}_{t}_{last_ts_value:.3f}_{targets[0]:.3f}_{targets[1]:.3f}_{targets[2]:.3f}_{targets[3]:.3f}_{targets[4]:.3f}.png'

def plot_spectrogram(coefficients, scales, sign_intensities, t, last_ts_value, targets, ticker):
    _, ax1 = plt.subplots(figsize=(1.28, 1.28))
    ax1.contourf(np.arange(sign_intensities.shape[1]), scales, np.abs(coefficients[::-1]), 300, cmap='jet')
    ax1.set_yscale('log')
    ax2 = ax1.inset_axes([0.0, 0.875, 1.0, 0.125])
    ax2.imshow(sign_intensities, aspect='auto', cmap='gray', vmin=0, vmax=255)
    ax2.axis('off')
    ax2.margins(0)
    plt.axis('off')
    plt.margins(0)
    plt.gca().set_position([0, 0, 1, 1])
    file_name = generate_file_name(t, last_ts_value, targets, ticker)
    try:
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
    except:
        os.mkdir(f'VolumeSpectrograms/{ticker}')
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()

def generate_spectrogram(data, start_idx, window_size, step_size, trading_days_skeleton, ticker):
    end_idx = start_idx + window_size
    step_idx = end_idx + step_size
    check = check_for_missing_values(data, start_idx, step_idx, trading_days_skeleton)
    if check:
        return
    t, ts, last_ts_value, targets = generate_time_series(data, start_idx, end_idx, step_idx)
    if np.max(ts) == np.min(ts):
        return
    ts = normalize_time_series(ts)
    sign_intensities = generate_sign_intensities(ts)
    scales = np.logspace(np.log10(0.1), np.log10(300), 500)
    coefficients = generate_continuous_wavelet_transform_coefficients(ts, scales)
    plot_spectrogram(coefficients, scales, sign_intensities, t, last_ts_value, targets, ticker)

TICKERS = pd.read_csv('current_tickers.csv')['0'].values.tolist()
TRADING_DAY_SKELETON = pd.read_csv('trading_days_skeleton.csv')
TRADING_DAY_SKELETON['Date'] = pd.to_datetime(TRADING_DAY_SKELETON['Date'])

def generate_spectrogram_mp(ticker):
    print('Working on:', ticker)
    data = getAdjCloseData(ticker, START_DATE, END_DATE)
    idx_adj = WINDOW_SIZE + STEP_SIZE
    indices = np.arange(0, data.shape[0] - idx_adj, 1)
    for start_idx in indices:
        generate_spectrogram(data, start_idx, WINDOW_SIZE, STEP_SIZE, TRADING_DAY_SKELETON, ticker)

def main():
    num_workers = 16
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(generate_spectrogram_mp, TICKERS)

if __name__ == '__main__':
    main()