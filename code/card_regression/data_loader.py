import torch
import os
import utils
import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler


def _get_index_train_test_path(data_directory_path, split_num, train=True):
    """
    Method to generate the path containing the training/test split for the given
    split number (generally from 1 to 20).
    @param split_num      Split number for which the data has to be generated
    @param train          Is true if the data is training data. Else false.
    @return path          Path of the file containing the requried data
    """
    if train:
        return os.path.join(data_directory_path, "index_train_" + str(split_num) + ".txt")
    else:
        return os.path.join(data_directory_path, "index_test_" + str(split_num) + ".txt")


def onehot_encode_cat_feature(X, cat_var_idx_list):
    """
    Apply one-hot encoding to the categorical variable(s) in the feature set,
        specified by the index list.
    """
    # select numerical features
    X_num = np.delete(arr=X, obj=cat_var_idx_list, axis=1)
    # select categorical features
    X_cat = X[:, cat_var_idx_list]
    X_onehot_cat = []
    for col in range(X_cat.shape[1]):
        X_onehot_cat.append(pd.get_dummies(X_cat[:, col], drop_first=True))
    X_onehot_cat = np.concatenate(X_onehot_cat, axis=1).astype(np.float32)
    dim_cat = X_onehot_cat.shape[1]  # number of categorical feature(s)
    X = np.concatenate([X_num, X_onehot_cat], axis=1)
    return X, dim_cat


def preprocess_uci_feature_set(X, config):
    """
    Obtain preprocessed UCI feature set X (one-hot encoding applied for categorical variable)
        and dimension of one-hot encoded categorical variables.
    """
    dim_cat = 0
    task_name = config.data.dir
    if config.data.one_hot_encoding:
        if task_name == "bostonHousing":
            X, dim_cat = onehot_encode_cat_feature(X, [3])
        elif task_name == "energy":
            X, dim_cat = onehot_encode_cat_feature(X, [4, 6, 7])
        elif task_name == "naval-propulsion-plant":
            X, dim_cat = onehot_encode_cat_feature(X, [0, 1, 8, 11])
        else:
            pass
    return X, dim_cat


############################
### UCI regression tasks ###
class UCI_Dataset(object):
    def __init__(self, config, split, validation=False):
        # global variables for reading data files
        _DATA_DIRECTORY_PATH = os.path.join(config.data.data_root, config.data.dir, "data")
        _DATA_FILE = os.path.join(_DATA_DIRECTORY_PATH, "data.txt")
        _INDEX_FEATURES_FILE = os.path.join(_DATA_DIRECTORY_PATH, "index_features.txt")
        _INDEX_TARGET_FILE = os.path.join(_DATA_DIRECTORY_PATH, "index_target.txt")
        _N_SPLITS_FILE = os.path.join(_DATA_DIRECTORY_PATH, "n_splits.txt")

        # set random seed 1 -- same setup as MC Dropout
        utils.set_random_seed(1)

        # load the data
        data = np.loadtxt(_DATA_FILE)
        # load feature and target indices
        index_features = np.loadtxt(_INDEX_FEATURES_FILE)
        index_target = np.loadtxt(_INDEX_TARGET_FILE)
        # load feature and target as X and y
        X = data[:, [int(i) for i in index_features.tolist()]].astype(np.float32)
        y = data[:, int(index_target.tolist())].astype(np.float32)
        # preprocess feature set X
        X, dim_cat = preprocess_uci_feature_set(X=X, config=config)
        self.dim_cat = dim_cat

        # load the indices of the train and test sets
        index_train = np.loadtxt(_get_index_train_test_path(_DATA_DIRECTORY_PATH, split, train=True))
        index_test = np.loadtxt(_get_index_train_test_path(_DATA_DIRECTORY_PATH, split, train=False))

        # read in data files with indices
        x_train = X[[int(i) for i in index_train.tolist()]]
        y_train = y[[int(i) for i in index_train.tolist()]].reshape(-1, 1)
        x_test = X[[int(i) for i in index_test.tolist()]]
        y_test = y[[int(i) for i in index_test.tolist()]].reshape(-1, 1)

        # split train set further into train and validation set for hyperparameter tuning
        if validation:
            num_training_examples = int(config.diffusion.nonlinear_guidance.train_ratio * x_train.shape[0])
            x_test = x_train[num_training_examples:, :]
            y_test = y_train[num_training_examples:]
            x_train = x_train[0:num_training_examples, :]
            y_train = y_train[0:num_training_examples]

        self.x_train = x_train if type(x_train) is torch.Tensor else torch.from_numpy(x_train)
        self.y_train = y_train if type(y_train) is torch.Tensor else torch.from_numpy(y_train)
        self.x_test = x_test if type(x_test) is torch.Tensor else torch.from_numpy(x_test)
        self.y_test = y_test if type(y_test) is torch.Tensor else torch.from_numpy(y_test)

        self.train_n_samples = x_train.shape[0]
        self.train_dim_x = self.x_train.shape[1]  # dimension of training data input
        self.train_dim_y = self.y_train.shape[1]  # dimension of training regression output

        self.test_n_samples = x_test.shape[0]
        self.test_dim_x = self.x_test.shape[1]  # dimension of testing data input
        self.test_dim_y = self.y_test.shape[1]  # dimension of testing regression output

        self.normalize_x = config.data.normalize_x
        self.normalize_y = config.data.normalize_y
        self.scaler_x, self.scaler_y = None, None

        if self.normalize_x:
            self.normalize_train_test_x()
        if self.normalize_y:
            self.normalize_train_test_y()

    def normalize_train_test_x(self):
        """
        When self.dim_cat > 0, we have one-hot encoded number of categorical variables,
            on which we don't conduct standardization. They are arranged as the last
            columns of the feature set.
        """
        self.scaler_x = StandardScaler(with_mean=True, with_std=True)
        if self.dim_cat == 0:
            self.x_train = torch.from_numpy(self.scaler_x.fit_transform(self.x_train).astype(np.float32))
            self.x_test = torch.from_numpy(self.scaler_x.transform(self.x_test).astype(np.float32))
        else:  # self.dim_cat > 0
            x_train_num, x_train_cat = self.x_train[:, : -self.dim_cat], self.x_train[:, -self.dim_cat :]
            x_test_num, x_test_cat = self.x_test[:, : -self.dim_cat], self.x_test[:, -self.dim_cat :]
            x_train_num = torch.from_numpy(self.scaler_x.fit_transform(x_train_num).astype(np.float32))
            x_test_num = torch.from_numpy(self.scaler_x.transform(x_test_num).astype(np.float32))
            self.x_train = torch.from_numpy(np.concatenate([x_train_num, x_train_cat], axis=1))
            self.x_test = torch.from_numpy(np.concatenate([x_test_num, x_test_cat], axis=1))

    def normalize_train_test_y(self):
        self.scaler_y = StandardScaler(with_mean=True, with_std=True)
        self.y_train = torch.from_numpy(self.scaler_y.fit_transform(self.y_train).astype(np.float32))
        self.y_test = torch.from_numpy(self.scaler_y.transform(self.y_test).astype(np.float32))

    def return_dataset(self, split="train"):
        if split == "train":
            train_dataset = torch.cat((self.x_train, self.y_train), dim=1)
            return train_dataset
        else:
            test_dataset = torch.cat((self.x_test, self.y_test), dim=1)
            return test_dataset

    def summary_dataset(self, split="train"):
        if split == "train":
            return {"n_samples": self.train_n_samples, "dim_x": self.train_dim_x, "dim_y": self.train_dim_y}
        else:
            return {"n_samples": self.test_n_samples, "dim_x": self.test_dim_x, "dim_y": self.test_dim_y}


def compute_y_noiseless_mean(dataset, x_test_batch, true_function="linear"):
    """
    Compute the mean of y with the ground truth data generation function.
    """
    if true_function == "linear":
        y_true_mean = dataset.a + dataset.b * x_test_batch
    elif true_function == "quadratic":
        y_true_mean = dataset.a * x_test_batch.pow(2) + dataset.b * x_test_batch + dataset.c
    elif true_function == "loglinear":
        y_true_mean = (dataset.a + dataset.b * x_test_batch).exp()
    elif true_function == "loglog":
        y_true_mean = (np.log(dataset.a) + dataset.b * x_test_batch.log()).exp()
    elif true_function == "mdnsinusoidal":
        y_true_mean = x_test_batch + 0.3 * torch.sin(2 * np.pi * x_test_batch)
    elif true_function == "sinusoidal":
        y_true_mean = x_test_batch * torch.sin(x_test_batch)
    else:
        raise NotImplementedError("We don't have such data generation scheme for toy example.")
    return y_true_mean.numpy()


class Renewable(object):
    def __init__(self, config, validation):
        dtype = config.data.type
        assert dtype in ["solar", "wind"], "Only solar and wind data available!"
        # Train set : always
        # Test set : validation - False
        # Validation set : validation - True
        # global variables for reading data files
        _DATA_DIRECTORY_PATH = os.path.join(config.data.data_root, config.data.dir)
        _DATA_FILE = os.path.join(_DATA_DIRECTORY_PATH, f"{dtype}.npy")
        _TARGET_FILE = os.path.join(_DATA_DIRECTORY_PATH, "Realised_Supply_Germany.csv")
        if validation:
            print("Loading train and validation data")
        else:
            print("Loading train and test data")
            _TEST_DATA_FILE = os.path.join(_DATA_DIRECTORY_PATH, f"{dtype}_2022.npy")
        # set random seed 1 -- same setup as MC Dropout
        utils.set_random_seed(1)

        x_data = np.load(_DATA_FILE)
        if dtype == "solar":
            y_data = pd.read_csv(_TARGET_FILE)["Photovoltaic [MW]"].values
        else:
            y_data = pd.read_csv(_TARGET_FILE)["Wind Total [MW]"].values
        if not validation:
            x_test_data = np.load(_TEST_DATA_FILE)
            y_test_data = y_data[(365 * 2 + 366) * 24 :][:, None]
        y_data = y_data[: (365 * 2 + 366) * 24][:, None]

        x_data_ = x_data.mean(1).reshape(x_data.shape[0], -1)
        if not validation:
            x_test_data_ = x_test_data.mean(1).reshape(x_test_data.shape[0], -1)

        self.normalize_x = config.data.normalize_x
        self.normalize_y = config.data.normalize_y
        self.scaler_x, self.scaler_y = None, None

        if self.normalize_x:
            x_data_ = self.normalize_x_(x_data_, True)
            if not validation:
                x_test_data_ = self.normalize_x_(x_test_data_)
        if self.normalize_y:
            y_data = self.normalize_y_(y_data, True)
            if not validation:
                y_test_data = self.normalize_y_(y_test_data)

        x_data_ = np.c_[
            x_data_,
            np.sin(np.array(range(len(x_data_))) * np.pi / 24),
            np.sin(np.array(range(len(x_data_))) * np.pi / (24 * 30)),
        ]
        if not validation:
            x_test_data_ = np.c_[
                x_test_data_,
                np.sin(np.array(range(len(x_test_data_))) * np.pi / 24),
                np.sin(np.array(range(len(x_test_data_))) * np.pi / (24 * 30)),
            ]

        if config.data.add_prev:
            x_data_ = np.hstack([x_data_, y_data])
            if not validation:
                x_test_data_ = np.hstack([x_test_data_, y_test_data])

        ws = config.data.window_size
        if config.data.hour_24:
            ws = ws + 24 - 1
            x_data_ = copy.deepcopy(np.lib.stride_tricks.sliding_window_view(x_data_, ws, 0).swapaxes(1, 2))[
                :, [*list(range(ws - 24)), -1]
            ]
            if not validation:
                x_test_data_ = copy.deepcopy(
                    np.lib.stride_tricks.sliding_window_view(x_test_data_, ws, 0).swapaxes(1, 2)
                )[:, [*list(range(ws - 24)), -1]]
        else:
            x_data_ = copy.deepcopy(np.lib.stride_tricks.sliding_window_view(x_data_, ws, 0).swapaxes(1, 2))
            if not validation:
                x_test_data_ = copy.deepcopy(
                    np.lib.stride_tricks.sliding_window_view(x_test_data_, ws, 0).swapaxes(1, 2)
                )

        if len(x_data_.shape) > 2:
            x_data_ = x_data_.reshape(x_data_.shape[0], -1)
            if not validation:
                x_test_data_ = x_test_data_.reshape(x_test_data_.shape[0], -1)

        # load feature and target as X and y
        X = x_data_.astype(np.float32)
        y = y_data.astype(np.float32)[ws - 1 :]
        if not validation:
            x_test_data = x_test_data_.astype(np.float32)
            y_test_data = y_test_data.astype(np.float32)[ws - 1 :]

        if validation:
            idx_split = (365 + 366) * 24
            x_train = X[:idx_split]
            y_train = y[:idx_split].reshape(-1, 1)
            x_test = X[idx_split:]
            y_test = y[idx_split:].reshape(-1, 1)
        else:
            x_train = X
            y_train = y.reshape(-1, 1)
            x_test = x_test_data
            y_test = y_test_data.reshape(-1, 1)

        self.x_train = x_train if type(x_train) is torch.Tensor else torch.from_numpy(x_train)
        self.y_train = y_train if type(y_train) is torch.Tensor else torch.from_numpy(y_train)
        self.x_test = x_test if type(x_test) is torch.Tensor else torch.from_numpy(x_test)
        self.y_test = y_test if type(y_test) is torch.Tensor else torch.from_numpy(y_test)

        self.train_n_samples = x_train.shape[0]
        self.train_dim_x = self.x_train.shape[1]  # dimension of training data input
        self.train_dim_y = self.y_train.shape[1]  # dimension of training regression output

        self.test_n_samples = x_test.shape[0]
        self.test_dim_x = self.x_test.shape[1]  # dimension of testing data input
        self.test_dim_y = self.y_test.shape[1]  # dimension of testing regression output

        if config.data.add_prev:
            self.x_train[:, -1] = 0
            self.x_test[:, -1] = 0

    def normalize_x_(self, x, fit=False):
        if fit:
            self.scaler_x = StandardScaler(with_mean=True, with_std=True)
            x = self.scaler_x.fit_transform(x).astype(np.float32)
        else:
            x = self.scaler_x.transform(x).astype(np.float32)
        return x

    def normalize_y_(self, y, fit=False):
        if fit:
            self.scaler_y = StandardScaler(with_mean=True, with_std=True)
            y = self.scaler_y.fit_transform(y).astype(np.float32)
        else:
            y = self.scaler_y.transform(y).astype(np.float32)
        return y

    def return_dataset(self, split="train"):
        if split == "train":
            train_dataset = torch.cat((self.x_train, self.y_train), dim=1)
            return train_dataset
        else:
            test_dataset = torch.cat((self.x_test, self.y_test), dim=1)
            return test_dataset

    def summary_dataset(self, split="train"):
        if split == "train":
            return {"n_samples": self.train_n_samples, "dim_x": self.train_dim_x, "dim_y": self.train_dim_y}
        else:
            return {"n_samples": self.test_n_samples, "dim_x": self.test_dim_x, "dim_y": self.test_dim_y}
