# Environment uses Python 3.10.17 and scikit-learn 1.6.1
import pickle
import warnings
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline

# Saved models and preprocessing objects live in <repo>/models/
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# Suppress FutureWarning from scikit-learn
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def hype_spectral_wrapper(i):
    # Code adapted from the method's notebook

    # For PRISMA, I remove bands affected by atmospheric water absorptiption for all target variables except B and Mn (indices 2 and 5)
    # Affected bands corresponding to the 1328–1523 nm and 1812–2044 nm spectral ranges
    atmospheric_bands = [i for i in range(97, 115)] + [i for i in range(144, 170)]

    # Raw PRISMA bands with high correlation to the corresponding target variable
    # Band 191 (2214 nm) was selected with intuition to help distinguish fraction of vegetation and bare soil in average PRISMA reflectance
    raw_hsi_bands_indices_dict = {
        "top_features_0": [191, 176, 219, 206, 46, 57, 0, 11],
        "top_features_1": [191, 2, 218, 143, 217, 161, 150],
        "top_features_2": [191, 0, 11, 98, 39, 78, 57, 143],
        "top_features_3": [191, 62, 76, 3, 14, 29, 40, 51],
        "top_features_4": [191, 2, 218, 143, 217, 161, 150],
        "top_features_5": [191, 0, 100, 111, 146, 161, 172, 228],
    }

    def transform_list(lst):
        # Calculate the new indices in the list of selected raw PRISMA bands after removing the bands affected by water absorption
        # Removal of water absorption bands was implemented after selecting the additional bands, hence this helper function
        transformed = []
    
        for num in lst:
            if num < 97:
                transformed.append(num)  # Index unchanged
            elif 97 <= num < 115 or 144 <= num < 170:
                continue  # drop from the list if band affected by atmospheric water absorption
            elif 115 <= num < 144:
                transformed.append(num - 18)  # 115 - 97 = 18
            else:  # num >= 170
                transformed.append(num - 44)  # 18 + (170 - 144) = 44
    
        return transformed
    
    def append_additional_features(X_hsi, raw_hsi_bands_indices):
        # Add unprocessed HSI bands as additional features
        X_raw_hsi_bands = []
    
        for idx in raw_hsi_bands_indices:
            X_hsi_idx_arr = X_hsi[:, idx].reshape(-1, 1)
            X_raw_hsi_bands.append(X_hsi_idx_arr)
            
        return np.hstack(X_raw_hsi_bands)


    # One Regressor for each target variable
    with open(MODELS_DIR / 'ExtraTrees_models_dict.pkl', 'rb') as f:
        best_models = pickle.load(f)
    
    # Indices of optimal set of features used during training and to be used for testing (obtained using Recursive Feature Elimination)
    with open(MODELS_DIR / 'best_features_indices_dict.pkl', 'rb') as f:
        best_features_indices_dict = pickle.load(f)
    
    with open(MODELS_DIR / 'msi_scaler.pkl', 'rb') as f:
        msi_scaler = pickle.load(f)
    
    with open(MODELS_DIR / 'pca_msi.pkl', 'rb') as f:
        pca_msi = pickle.load(f)
    
    with open(MODELS_DIR / 'pca_hsi.pkl', 'rb') as f:
        pca_hsi = pickle.load(f)
    
    with open(MODELS_DIR / 'pca_hsi_filtered.pkl', 'rb') as f:
        pca_hsi_filtered = pickle.load(f)

    # Custom transformer to remove atmospheric bands from HSI data
    class AtmosphericBandsRemover:
        def __init__(self, atmospheric_bands):
            self.atmospheric_bands = atmospheric_bands
    
        def fit(self, X, y=None):
            return self
    
        def transform(self, X):
            return np.delete(X, self.atmospheric_bands, axis=1)
    
    # Custom transformer for feature selection
    class FeatureSelector:
        def __init__(self, i):
            self.feature_indices = best_features_indices_dict[f'target_{i}']
    
        def fit(self, X, y=None):
            return self
    
        def transform(self, X):
            return np.take(X, self.feature_indices, axis=1)
    
    # Custom transformer for additional feature extraction from HSI
    class AdditionalFeatureExtractor:
        def __init__(self, raw_hsi_bands_indices_dict, i):
            self.raw_hsi_bands_indices_dict = raw_hsi_bands_indices_dict
            self.i = i
    
        def fit(self, X, y=None):
            return self
    
        def transform(self, X):
            if self.i in [2, 5]: # B or Mn
                additional_features = append_additional_features(X, self.raw_hsi_bands_indices_dict["top_features_" + str(self.i)])

            else:
                transformed_list = transform_list(self.raw_hsi_bands_indices_dict["top_features_" + str(self.i)])
                additional_features = append_additional_features(X, transformed_list)
            return additional_features
    
    # Define a preprocessing pipeline for MSI data
    msi_pipeline = Pipeline([
        ('scaler', msi_scaler),
        ('pca', pca_msi)
    ])
            
    model = best_models[f'target_{i+1}']

    if i in [2, 5]:
        # No removal of bands affected by water absorption for B or Mn
        # Reason: were already outperforming other methods during the challenge
        hsi_pipeline = Pipeline([
            ('pca', pca_hsi)
        ])
        addfeats_pipeline = Pipeline([
        ('additional_feature_extractor', AdditionalFeatureExtractor(raw_hsi_bands_indices_dict, i))
        ])
    else:
        # Define a preprocessing pipeline for HSI data (including filtered bands)
        hsi_pipeline = Pipeline([
            ('atmospheric_bands_remover', AtmosphericBandsRemover(atmospheric_bands)),
            ('pca_filtered', pca_hsi_filtered),
        ])
        addfeats_pipeline = Pipeline([
            ('atmospheric_bands_remover', AtmosphericBandsRemover(atmospheric_bands)),
            ('additional_feature_extractor', AdditionalFeatureExtractor(raw_hsi_bands_indices_dict, i))
        ])

    featselector_pipeline = Pipeline([
        ('feature_selector', FeatureSelector(i))
    ])
    
    def predict_with_pipeline(X):
        X_msi_raw, X_hsi_raw = X[:, :12], X[:, 12:]            
        
        return model.predict(
            featselector_pipeline.transform(
                np.hstack((
                    hsi_pipeline.transform(X_hsi_raw),
                    msi_pipeline.transform(X_msi_raw),
                    addfeats_pipeline.transform(X_hsi_raw)
                ))
            )
        )

    return predict_with_pipeline
            