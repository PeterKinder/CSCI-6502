import pandas as pd
import numpy as np
import os
import shutil
import time
from PIL import Image
import multiprocessing

def parse_filename(filename, step):
    values = filename.split('_')
    date = pd.to_datetime(values[1])
    last_price = values[2]
    if step == 1:
        target = values[3]
    if step == 5:
        target = values[4]
    if step == 10:
        target = values[5]
    if step == 15:
        target = values[6]
    if step == 20:
        target = values[7]
    if target > last_price:
        target = 1
    else:
        target = 0
    return date, target

def get_adjPrice_dates(ticker):
    all_files = []
    files = os.listdir(f'AdjCloseSpectrograms/{ticker}')
    all_files += files
    training_set_file_1 = []
    training_set_file_0 = []
    for filename in all_files:
        date, target = parse_filename(filename, 20)
        if target == 1:
            training_set_file_1.append("_".join(filename.split('_')[:2]))
        else:
            training_set_file_0.append("_".join(filename.split('_')[:2]))
    return training_set_file_1, training_set_file_0

def get_volume_dates(ticker):
    all_volume_files = []
    files = os.listdir(f'VolumeSpectrograms/{ticker}')
    files = ["_".join(file.split('_')[:2]) for file in files]
    all_volume_files += files
    return all_volume_files

def get_adjPrice_file(date, all_adj_files):
    for file in all_adj_files:
        if date in file:
            return f'AdjCloseSpectrograms/{file.split("_")[0]}/{file}'

def get_Volume_file(date, all_volume_files):
    for file in all_volume_files:
        if date in file:
            return f'VolumeSpectrograms/{file.split("_")[0]}/{file}'

def get_SP500_file(date, all_SP500_files):
    for file in all_SP500_files:
        if date.split('_')[1] in file:
            return f'AdjCloseSpectrograms/SP500/{file}'

def get_VIX_file(date, all_VIX_files):
    for file in all_VIX_files:
        if date.split('_')[1] in file:
            return f'AdjCloseSpectrograms/VIX/{file}'

def prepare_data(ticker):
    try:
        print(f'Preparing data for {ticker}')
        trading_days_skeleton = pd.read_csv('trading_days_skeleton.csv')
        trading_days_skeleton['Date'] = pd.to_datetime(trading_days_skeleton['Date'])
        sp500_ticker_start_end = pd.read_csv('sp500_ticker_start_end.csv')
        sp500_ticker_start_end['start_date'] = pd.to_datetime(sp500_ticker_start_end['start_date'])
        sp500_ticker_start_end['end_date'] = pd.to_datetime(sp500_ticker_start_end['end_date'])
        sp500_ticker_start_end['end_date'] = sp500_ticker_start_end['end_date'].fillna(pd.to_datetime('2025-04-01'))
        trading_days_skeleton['year'] = trading_days_skeleton['Date'].dt.year
        adjClose_dates_1, adjClose_dates_0 = get_adjPrice_dates(ticker)
        volume_dates = get_volume_dates(ticker)
        adjClose_dates_1_intersection = list(set(adjClose_dates_1).intersection(set(volume_dates)))
        adjClose_dates_0_intersection = list(set(adjClose_dates_0).intersection(set(volume_dates)))
        all_adj_files = os.listdir(f'AdjCloseSpectrograms/{ticker}')
        all_volume_files = os.listdir(f'VolumeSpectrograms/{ticker}')
        all_SP500_files = os.listdir('AdjCloseSpectrograms/SP500')
        all_VIX_files = os.listdir('AdjCloseSpectrograms/VIX')
        all_adj_files = [file for file in all_adj_files if pd.to_datetime(file.split('_')[1]) > pd.to_datetime('2024-06-01') and pd.to_datetime(file.split('_')[1]) < pd.to_datetime('2025-04-01')]
        all_volume_files = [file for file in all_volume_files if pd.to_datetime(file.split('_')[1]) > pd.to_datetime('2024-06-01') and pd.to_datetime(file.split('_')[1]) < pd.to_datetime('2025-04-01')]
        all_SP500_files = [file for file in all_SP500_files if pd.to_datetime(file.split('_')[1]) > pd.to_datetime('2024-06-01') and pd.to_datetime(file.split('_')[1]) < pd.to_datetime('2025-04-01')]
        all_VIX_files = [file for file in all_VIX_files if pd.to_datetime(file.split('_')[1]) > pd.to_datetime('2024-06-01') and pd.to_datetime(file.split('_')[1]) < pd.to_datetime('2025-04-01')]
        train_1_dict = {date: [
            get_adjPrice_file(date, all_adj_files),
            get_Volume_file(date, all_volume_files),
            get_SP500_file(date, all_SP500_files),
            get_VIX_file(date, all_VIX_files)
        ] for date in adjClose_dates_1_intersection}
        train_0_dict = {date: [
            get_adjPrice_file(date, all_adj_files),
            get_Volume_file(date, all_volume_files),
            get_SP500_file(date, all_SP500_files),
            get_VIX_file(date, all_VIX_files)
        ] for date in adjClose_dates_0_intersection}
        for date, files in train_1_dict.items():
            try:
                for idx, file in enumerate(files):
                    image = Image.open(file)
                    if idx == 0:
                        combined_image = np.array(image.convert('RGB'))
                    else:
                        combined_image = np.concatenate((combined_image, np.array(image.convert('RGB'))), axis=2)
                combined_image = np.transpose(combined_image, (2, 0, 1))
                np.save(f'TrainingData/ALL/multipleTimeSeriesData/{date}_1.npy', combined_image)
            except:
                pass
        for date, files in train_0_dict.items():
            try:
                for idx, file in enumerate(files):
                    image = Image.open(file)
                    if idx == 0:
                        combined_image = np.array(image.convert('RGB'))
                    else:
                        combined_image = np.concatenate((combined_image, np.array(image.convert('RGB'))), axis=2)
                combined_image = np.transpose(combined_image, (2, 0, 1))
                np.save(f'TrainingData/ALL/multipleTimeSeriesData/{date}_0.npy', combined_image)
            except:
                pass
    except Exception as e:
        print(f'Error preparing data for {ticker}: {e}')

TICKERS = os.listdir(f'AdjCloseSpectrograms')

def main():
    num_workers = 16
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(prepare_data, TICKERS)

if __name__ == '__main__':
    main()