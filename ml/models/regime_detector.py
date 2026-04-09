"""Gaussian Hidden Markov Model for market regime detection."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings("ignore")


class GaussianHMMRegimeDetector:
    """
    Detects market regimes using Gaussian HMM on price + macro features.
    
    Attributes:
        n_regimes: Number of hidden regimes to detect
        pca_components: Number of PCA components for dimensionality reduction
        model: Fitted GaussianHMM instance
        scaler: StandardScaler for feature normalization
        pca: PCA instance for dimensionality reduction
    """
    
    def __init__(self, n_regimes: int = 4, pca_components: int = 10, random_state: int = 42):
        """
        Initialize regime detector.
        
        Args:
            n_regimes: Number of hidden market regimes (typically 3-4)
            pca_components: Number of PCA components for preprocessing
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.pca_components = pca_components
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components, random_state=random_state)
        
        # Store regime statistics
        self.regime_stats = None
        self.regime_names = None
        
    def fit(self, 
            features: pd.DataFrame, 
            regime_names: List[str] = None) -> "GaussianHMMRegimeDetector":
        """
        Fit HMM to features and identify regime characteristics.
        
        Args:
            features: DataFrame with features (weekly observations)
            regime_names: Optional names for regimes (e.g., ["Risk-On", "Defensive", "Panic"])
            
        Returns:
            self for method chaining
        """
        # Extract values if DataFrame (reset index to ensure pure numeric data)
        if isinstance(features, pd.DataFrame):
            X = features.reset_index(drop=True).values.astype(np.float64)
        else:
            X = np.asarray(features, dtype=np.float64)
            
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA reduction
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Fit HMM
        self.model = GaussianHMM(n_components=self.n_regimes, 
                                  covariance_type="diag",
                                  n_iter=1000,
                                  random_state=self.random_state)
        self.model.fit(X_pca)
        
        # Compute regime statistics
        self._compute_regime_stats(features)
        
        # Set regime names
        if regime_names is not None:
            assert len(regime_names) == self.n_regimes
            self.regime_names = regime_names
        else:
            self.regime_names = [f"Regime_{i}" for i in range(self.n_regimes)]
            
        return self
    
    def predict_regimes(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict regime labels for new features.
        
        Args:
            features: DataFrame with shape (n_obs, n_features)
            
        Returns:
            Array of regime labels (0 to n_regimes-1), shape (n_obs,)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call .fit() first.")
        
        # Extract values if DataFrame
        if isinstance(features, pd.DataFrame):
            X = features.reset_index(drop=True).values.astype(np.float64)
        else:
            X = np.asarray(features, dtype=np.float64)
            
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.model.predict(X_pca)
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict regime probabilities (posterior) for new features.
        
        Args:
            features: DataFrame with shape (n_obs, n_features)
            
        Returns:
            Array of regime posteriors, shape (n_obs, n_regimes)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call .fit() first.")
        
        # Extract values if DataFrame
        if isinstance(features, pd.DataFrame):
            X = features.reset_index(drop=True).values.astype(np.float64)
        else:
            X = np.asarray(features, dtype=np.float64)
            
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.model.predict_proba(X_pca)
    
    def _compute_regime_stats(self, features: pd.DataFrame):
        """Compute average feature values for each regime."""
        regime_labels = self.predict_regimes(features)
        
        stats = {}
        for regime_id in range(self.n_regimes):
            mask = regime_labels == regime_id
            if mask.sum() > 0:
                stats[regime_id] = features[mask].mean().to_dict()
            else:
                stats[regime_id] = {}
                
        self.regime_stats = stats
    
    def get_regime_stats(self) -> Dict:
        """Return statistics for each regime."""
        return self.regime_stats
    
    def get_regime_names(self) -> List[str]:
        """Return regime names."""
        return self.regime_names
    
    def score(self, features: pd.DataFrame) -> float:
        """
        Compute log-likelihood score.
        
        Args:
            features: DataFrame with shape (n_obs, n_features)
            
        Returns:
            Log-likelihood score
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call .fit() first.")
            
        X_scaled = self.scaler.transform(features)
        X_pca = self.pca.transform(X_scaled)
        return self.model.score(X_pca)
