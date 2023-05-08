import numpy as np
from deapcalc import *
from sklearn import preprocessing
from velocity import velocityatdistance


def hybridmodeldataprep(raw_data):
    """
    :parameter
    np_data: A numpy array that consist training data for each bullet.

    :return: numpy array that include all features and difference value of training data and EoM, ready to be trained.
    """
    global scaler  # for inverse transform of the final result
    # df_the = raw_data.iloc[:, [0, 3, 4, 5, 9]]
    # np_data_the = df_the.to_numpy()

    df_data = raw_data.iloc[:, [0, 3, 4, 5, 6, 7, 8, 9]]

    df_for_hybrid = df_data.to_numpy()

    np_diff = df_for_hybrid.copy()

    sum_ = []
    v_this = []

    # for each bullet at the distance:
    for i in range(df_for_hybrid.shape[0]):
        row = df_for_hybrid[i]
        if row[0] == 0:
            v_final = row[6]
        else:
            v, d = velocityatdistance(
                ini_v=np.array([row[6], 0, 0]),
                distance=row[0],
                bc=row[1],
                mass=row[2],
                diameter=0.022033333333,
            )

            v_final = v[-1]
        sum_.append(v_final - row[-1])
        v_this.append(v_final)

    difference_col = np.transpose(np.asarray([sum_]))

    np_diff = np.delete(np_diff, -1, 1)
    np_diff = np.append(np_diff, np.transpose(np.asarray([v_this])), axis=1)
    np_diff = np.append(np_diff, difference_col, axis=1)

    # np_diff_whole = np.append(df_for_hybrid, np_diff[:, [5, 6]], axis=1)

    # np_diff_scale_del = np.delete(np_diff_whole, 7, 1)
    # np_diff_scale_del = np.delete(np_diff, 7, 1)
    scaler = preprocessing.MaxAbsScaler()
    np_diff_scale = scaler.fit_transform(np_diff)

    np_train = np_diff_scale[:, :-1]
    np_target = np_diff_scale[:, -1]

    return np_train, np_target, scaler


def func_hyb(hof, D, BC, Weight, boattail, roundtip, cannelure, IV, Veom):
    return eval(str(hof))

def hybridmodeltest(raw_data, scaler, hof):
    """
    :parameter
    np_data: A numpy array that consist training data for each bullet.

    :return: numpy array that include all features and difference value of training data and EoM, ready to be trained.
    """

    df_data = raw_data.iloc[:, [0, 3, 4, 5, 6, 7, 8, 9, 1]]

    df_for_hybrid1 = df_data.to_numpy()

    df_for_hybrid1 = np.delete(df_for_hybrid1, -1, axis=1)

    sum_ = []
    v_this = []

    # for each bullet at the distance:
    for i in range(df_for_hybrid1.shape[0]):
        row = df_for_hybrid1[i]
        if row[0] == 0:
            v_final = row[6]
        else:
            v, d = velocityatdistance(
                ini_v=np.array([row[6], 0, 0]),
                distance=row[0],
                bc=row[1],
                mass=row[2],
                diameter=0.022033333333,
            )

            v_final = v[-1] + scaler.scale_[-1] * func_hyb(
                hof,
                row[0] / scaler.scale_[0],
                row[1] / scaler.scale_[1],
                row[2] / scaler.scale_[2],
                row[3] / scaler.scale_[3],
                row[4] / scaler.scale_[4],
                row[5] / scaler.scale_[5],
                row[6] / scaler.scale_[6],
                v[-1] / scaler.scale_[-2],
            )
        sum_.append(v_final - row[-1])
        v_this.append(v_final)

    v_pred = np.transpose(np.asarray(v_this))
    v_real = df_for_hybrid1[:, -1]

    return v_pred, v_real
