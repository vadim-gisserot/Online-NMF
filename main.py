from time_series_ONMF_COVID19 import time_series_tensor
import numpy as np
from PIL import Image
from skimage.transform import downscale_local_mean
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.decomposition import SparseCoder
import itertools
from time import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

def preprocess_data(data_path, countries, cutoff_date):
    df = pd.read_csv(data_path)
    date_columns = list(df.columns[4:])
    selected_dates = date_columns[:date_columns.index(cutoff_date) + 1]

    return df.loc[
        df["Country/Region"].isin(countries),
        list(df.columns[:4]) + selected_dates,
    ].copy()

def main_train_joint():
    print('pre-training temporal dictionary started...')

    confirmed_path = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    deaths_path = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
    recovered_path = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

    foldername = 'reproduction_exp_1000'
    source = [confirmed_path, deaths_path, recovered_path]

    n_components = 50

    cutoff_date = "4/12/20" # Paper's dates: from Jan 19, 2020 to Apr. 12, 2020 -> missing Jan 19, 20, 21
    country_list = ['Korea, South', 'China', 'US', 'Italy', 'Germany', 'Spain']

    data_dir = Path("Time_series_dictionary") / foldername
    data_dir.mkdir(parents=True, exist_ok=True)

    filtered_source = []
    for original_path in source:
        df = preprocess_data(original_path, country_list, cutoff_date)
        output_path = data_dir / Path(original_path).name
        df.to_csv(output_path, index=False)
        filtered_source.append(str(output_path))

    source = filtered_source

    reconstructor = time_series_tensor(path=None,
                                       source=source,
                                       country_list=country_list,
                                       alpha=3,  # L1 sparsity regularizer
                                       beta=1,  # default learning exponent --
                                       # customized in both trianing and online prediction functions
                                       # learning rate exponent in online learning -- smaller weighs new data more
                                       n_components=n_components,  # number of dictionary elements -- rank
                                       iterations=20,  # number of iterations for the ONTF algorithm
                                       sub_iterations=2,  # number of i.i.d. subsampling for each iteration of ONTF
                                       batch_size=50,  # number of patches used in i.i.d. subsampling
                                       num_patches_perbatch=100,
                                       # number of patches per ONMF iteration (size of mini batch)
                                       # number of patches that ONTF algorithm learns from at each iteration
                                       patch_size=6,
                                       prediction_length=1,
                                       learnevery=1,
                                       subsample=False,
                                       if_onlynewcases=True,   # take the derivate of the time-series of total to get new cases
                                       if_moving_avg_data=True,
                                       if_log_scale=True)

    L = 30
    np.random.seed(42)
    avg_iter = 1000 # To try first
    A_recons = []
    # W_old = W.copy()
    for step in np.arange(avg_iter):

        ### Clear up the fields
        reconstructor.W = None
        reconstructor.code = np.zeros(shape=(n_components, 100))

        ### Minibatch dictionary learning
        W, At, Bt, H = reconstructor.train_dict(mode=3,
                                                foldername=foldername,
                                                alpha=1,  ## L1 regularizer for sparse coding
                                                beta=1,  ## learning rate exponent in pre-learning
                                                learn_joint_dict=True)

        ### Online dictionary learning and predictoin
        A_predict, error, W1, At1, Bt1, H = reconstructor.online_learning_and_prediction(mode=3,
                                                                                         ini_dict=W,
                                                                                         foldername=foldername,
                                                                                         beta=4,  # no effect if "if_learn_online" is false
                                                                                         ini_A=At,
                                                                                         ini_B=Bt,
                                                                                         a1=0, # regularizer for training
                                                                                         a2=0, # regularizer for prediction
                                                                                         future_extraploation_length=L,
                                                                                         if_learn_online = True,
                                                                                         if_save=True)

        A_recons.append(A_predict.tolist())
        print('Current iteration %i out of %i' % (step, avg_iter))

    A_recons = np.asarray(A_recons)
    np.save('Time_series_dictionary/' + str(foldername) + '/recons_nononline', A_recons)

    # print('change in dictionary after online learning', np.linalg.norm(W_old - W1))

    '''
    ### For loading saved checkpoints just for plotting
    W1 = np.load('Time_series_dictionary/' + str(foldername) + '/dict_learned_3_pretraining_Korea, South.npy')
    code = np.load('Time_series_dictionary/' + str(foldername) + '/code_learned_3_pretraining_Korea, South.npy')
    reconstructor.code = code
    A_recons = np.load("Time_series_dictionary/" + str(foldername) + "/recons_nononline.npy")
    '''

    ### plot minibatch-trained dictionary (from last iteration)
    reconstructor.display_dictionary(W, cases='confirmed', if_show=True, if_save=True, foldername=foldername, filename='minibatch')
    reconstructor.display_dictionary(W, cases='death', if_show=True, if_save=True, foldername=foldername, filename='minibatch')
    reconstructor.display_dictionary(W, cases='recovered', if_show=True, if_save=True, foldername=foldername, filename='minibatch')

    ### plot minibatch-then-online-trained dictionary (from last iteration)
    reconstructor.display_dictionary(W1, cases='confirmed', if_show=True, if_save=True, foldername=foldername, filename='online')
    reconstructor.display_dictionary(W1, cases='death', if_show=True, if_save=True, foldername=foldername, filename='online')
    reconstructor.display_dictionary(W1, cases='recovered', if_show=True, if_save=True, foldername=foldername, filename='online')

    ### plot original and prediction curves
    reconstructor.display_prediction(source, A_recons, cases='confirmed', if_show=True, if_save=True, foldername=foldername, if_errorbar=True)
    reconstructor.display_prediction(source, A_recons, cases='death', if_show=True, if_save=True, foldername=foldername, if_errorbar=True)
    reconstructor.display_prediction(source, A_recons, cases='recovered', if_show=True, if_save=True, foldername=foldername, if_errorbar=True)

def main():

    main_train_joint()

if __name__ == '__main__':
    main()

