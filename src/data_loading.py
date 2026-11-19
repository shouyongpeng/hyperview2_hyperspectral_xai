import numpy as np
import pandas as pd


class BaselineRegressor:
    """
    Baseline regressor, which calculates the mean value of the target from the training
    data and returns it for each testing sample.
    """

    def __init__(self):
        self.mean = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.mean = np.mean(y_train, axis=0)
        self.classes_count = y_train.shape[1]
        return self

    def predict(self, X_test: np.ndarray):
        return np.full((len(X_test), self.classes_count), self.mean)


def load_data_one_modality(directory, modality, reductions=['mean']):
    files = sorted(directory.glob('*.npz'))
    data_list = []
    n_valid_pixels_list = []
    for file in enumerate(files):
        file_name = file[1]
        with np.load(file_name) as npz:
            arr = np.ma.MaskedArray(**npz)
            
            # Get the number of valid pixels (non-masked elements)
            n_valid_pixels_sample = arr[0, :, :].count()
            n_valid_pixels_list.append(n_valid_pixels_sample)
            if modality == 'hsi_satellite':
                arr = arr.data

            if 'mean' in reductions:
                mean_pixel = np.mean(arr, axis=(1, 2))  # mean over all pixels
                data_list.append(mean_pixel)
            elif 'median' in reductions:
                mean_pixel = np.median(arr, axis=(1, 2))
                data_list.append(mean_pixel)
                
            else:  # No reduction
                data_list.append(arr)

    return data_list, n_valid_pixels_list
    

def load_data(reductions=['mean'], DATASET_DIR='HYPERVIEW2'):
    modalities = ['hsi_airborne', 'hsi_satellite', 'msi_satellite']

    X_train_dict = {}
    X_test_dict = {}
    n_valid_pixels_train_dict = {}
    n_valid_pixels_test_dict = {}

    # Load data
    for modality in modalities:
        data_modality, n_valid_pixels_list = load_data_one_modality(DATASET_DIR / 'train' / modality,
                                                                    modality,
                                                                    reductions)
        X_train_dict[modality] = data_modality
        n_valid_pixels_train_dict[modality] = n_valid_pixels_list

    for modality in modalities:
        if not modality == 'hsi_airborne':  # no test data for HSI airborne
            data_modality, n_valid_pixels_list = load_data_one_modality(DATASET_DIR / 'test' / modality,
                                                                        modality,
                                                                        reductions)
            X_test_dict[modality] = data_modality
            n_valid_pixels_test_dict[modality] = n_valid_pixels_list

    # Load the ground truth measurements
    GT_TRAIN_CSV_PATH = DATASET_DIR / 'train_gt.csv'
    gt_train_df = pd.read_csv(GT_TRAIN_CSV_PATH)
    gt_train_df.head()
    column_names = ['Fe', 'Zn', 'B', 'Cu', 'S', 'Mn']
    y_train_all = gt_train_df[column_names].values

    return X_train_dict, X_test_dict, y_train_all, n_valid_pixels_train_dict, n_valid_pixels_test_dict
