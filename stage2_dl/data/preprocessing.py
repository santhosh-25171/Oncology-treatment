import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# IMAGE PREPROCESSING
# -----------------------------------------------------------------------------
class ImagePreprocessor:
    def __init__(self, target_size=(128, 128), normalize=True):
        """
        target_size: Desired spatial dimension for CNN input (Default: 128x128).
                     NOTE: The original BreastMNIST dataset is 28x28. 
                     Resizing to 128x128 is strictly for CNN architectural compatibility.
                     It uses interpolation (bicubic/bilinear) and does NOT recover 
                     or artificially invent missing medical detail.
        normalize:   Scales pixel values to [0, 1].
        """
        self.target_size = target_size
        self.normalize = normalize
        
    def preprocess(self, img_path: str) -> np.ndarray:
        """
        Loads a RAW 28x28 image, interpolates it to PREPROCESSED CNN INPUT (e.g., 1x128x128),
        and normalizes it.
        """
        # Load grayscale
        img = Image.open(img_path).convert('L')
        
        # Bicubic interpolation from 28x28 to target_size (e.g. 128x128)
        img = img.resize(self.target_size, resample=Image.BICUBIC)
        
        img_array = np.array(img, dtype=np.float32)
        
        # Add channel dimension (C, H, W) for standard PyTorch-like tensors -> (1, 128, 128)
        img_array = np.expand_dims(img_array, axis=0)
        
        if self.normalize:
            # Independent scaling per image (does not leak test statistics)
            img_array = img_array / 255.0
            
        return img_array

# -----------------------------------------------------------------------------
# SEQUENCE PREPROCESSING
# -----------------------------------------------------------------------------
class SequencePreprocessor:
    def __init__(self):
        self.feature_means = None
        self.feature_stds = None
        
    def fit(self, X_train: np.ndarray) -> None:
        """
        Fit normalization statistics strictly on the TRAINING data.
        X_train shape: (num_samples, seq_len, num_features)
        """
        # Flatten temporal dimension to compute feature-wise mean/std
        flat_x = X_train.reshape(-1, X_train.shape[-1])
        # Ignore NaNs during fit
        self.feature_means = np.nanmean(flat_x, axis=0)
        self.feature_stds = np.nanstd(flat_x, axis=0)
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features using statistics derived exclusively from training data.
        Missing values (NaNs) are imputed to 0.0, which corresponds to the post-scaling mean.
        """
        if self.feature_means is None or self.feature_stds is None:
            raise RuntimeError("SequencePreprocessor must be fitted on training data before transform.")
            
        X_scaled = (X - self.feature_means) / (self.feature_stds + 1e-8)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        return X_scaled
