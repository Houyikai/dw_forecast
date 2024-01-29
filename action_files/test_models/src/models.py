import os
import time

import fire
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from src.core import NeuralForecast

from src.models.gru import GRU
from src.models.rnn import RNN
from src.models.tcn import TCN
from src.models.lstm import LSTM
from src.models.dilated_rnn import DilatedRNN
from src.models.deepar import DeepAR
from src.models.mlp import MLP
from src.models.nhits import NHITS
from src.models.nbeats import NBEATS
from src.models.nbeatsx import NBEATSx
from src.models.tft import TFT
from src.models.vanillatransformer import VanillaTransformer
from src.models.informer import Informer
from src.models.autoformer import Autoformer
from src.models.patchtst import PatchTST

from src.auto import (
    AutoMLP, AutoNHITS, AutoNBEATS, AutoDilatedRNN, AutoTFT
)

from src.losses.pytorch import SMAPE, MAE
from ray import tune

from src.data import get_data


def main(dataset: str = 'M3', group: str = 'Monthly') -> None:
    train, horizon, freq, seasonality = get_data('data/', dataset, group)
    train['ds'] = pd.to_datetime(train['ds'])

    config_nbeats = {
        "input_size": tune.choice([2 * horizon]),
        "max_steps": 1000,
        "val_check_steps": 300,
        "scaler_type": "minmax1",
        "random_seed": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    }
    config = {
        "hidden_size": tune.choice([256, 512]),
        "num_layers": tune.choice([2, 4]),
        "input_size": tune.choice([2 * horizon]),
        "max_steps": 1000,
        "val_check_steps": 300,
        "scaler_type": "minmax1",
        "random_seed": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    }
    config_drnn = {'input_size': tune.choice([2 * horizon]),
                   'encoder_hidden_size': tune.choice([124]),
                   "max_steps": 300,
                   "val_check_steps": 100,
                   "random_seed": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),}
    models = [
        AutoDilatedRNN(h=horizon, loss=MAE(), config=config_drnn, num_samples=2, cpus=1),
        RNN(h=horizon, input_size=2 * horizon, encoder_hidden_size=50, max_steps=300),
        TCN(h=horizon, input_size=2 * horizon, encoder_hidden_size=20, max_steps=300),
        NHITS(h=horizon, input_size=2 * horizon, dropout_prob_theta=0.5, loss=MAE(), max_steps=1000, val_check_steps=500),
        AutoMLP(h=horizon, loss=MAE(), config=config, num_samples=2, cpus=1),
        TFT(h=horizon, input_size=2 * horizon, loss=SMAPE(), hidden_size=64, scaler_type='robust', windows_batch_size=512, max_steps=1500, val_check_steps=500),
        VanillaTransformer(h=horizon, input_size=2 * horizon, loss=MAE(), hidden_size=64, scaler_type='minmax1', windows_batch_size=512, max_steps=1500, val_check_steps=500),
        DeepAR(h=horizon, input_size=2 * horizon, scaler_type='minmax1', max_steps=1000),
    ]

    # Models
    for model in models[:-1]:
        model_name = type(model).__name__
        print(50*'-', model_name, 50*'-')
        start = time.time()
        fcst = NeuralForecast(models=[model], freq=freq)
        fcst.fit(train)
        forecasts = fcst.predict()
        end = time.time()
        print(end - start)

        forecasts = forecasts.reset_index()
        forecasts.columns = ['unique_id', 'ds', model_name]
        forecasts.to_csv(f'data/{model_name}-forecasts-{dataset}-{group}.csv', index=False)
        time_df = pd.DataFrame({'time': [end - start], 'model': [model_name]})
        time_df.to_csv(f'data/{model_name}-time-{dataset}-{group}.csv', index=False)

    # DeepAR
    model_name = type(models[-1]).__name__
    start = time.time()
    fcst = NeuralForecast(models=[models[-1]], freq=freq)
    fcst.fit(train)
    forecasts = fcst.predict()
    end = time.time()
    print(end - start)

    forecasts = forecasts.reset_index()
    forecasts = forecasts[['unique_id', 'ds', 'DeepAR-median']]
    forecasts.columns = ['unique_id', 'ds', 'DeepAR']
    forecasts.to_csv(f'data/{model_name}-forecasts-{dataset}-{group}.csv', index=False)
    time_df = pd.DataFrame({'time': [end - start], 'model': [model_name]})
    time_df.to_csv(f'data/{model_name}-time-{dataset}-{group}.csv', index=False)


if __name__ == '__main__':
    fire.Fire(main)