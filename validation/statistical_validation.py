"""
Enhanced statistical validation with Deflated Sharpe Ratios and advanced cross-validation
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from scipy import stats
from scipy.optimize import minimize_scalar
import logging

log = logging.getLogger(__name__)

class DeflatedSharpeRatio:
    """
    Deflated Sharpe Ratio (DSR) - Accounts for multiple testing and selection bias
    Based on Marcos López de Prado's work on avoiding overfitting in backtests
    """
    
    def __init__(self, benchmark_sharpe: float = 0.0):
        self.benchmark_sharpe = benchmark_sharpe
        
    def compute_deflated_sharpe(self, returns: pd.Series, n_trials: int = 1, 
                              variance_estimate: Optional[float] = None) -> Dict[str, float]:
        """
        Compute Deflated Sharpe Ratio
        
        Args:
            returns: Strategy returns
            n_trials: Number of trials/strategies tested (for multiple testing adjustment)
            variance_estimate: Variance of Sharpe ratio estimate (optional)
            
        Returns:
            Dictionary with DSR metrics
        """
        if len(returns) < 30:
            log.warning("Insufficient data for reliable DSR calculation")
            return {'dsr': 0.0, 'psr': 0.0, 'sharpe': 0.0, 'p_value': 1.0}
        
        # Calculate standard Sharpe ratio
        returns_clean = returns.dropna()
        mean_ret = returns_clean.mean()
        std_ret = returns_clean.std()
        
        if std_ret == 0:
            return {'dsr': 0.0, 'psr': 0.0, 'sharpe': 0.0, 'p_value': 1.0}
            
        sharpe = mean_ret / std_ret * np.sqrt(252)  # Annualized
        
        # Probabilistic Sharpe Ratio (PSR)
        n_obs = len(returns_clean)
        
        if variance_estimate is None:
            # Estimate variance of Sharpe ratio under normal returns
            # Var(SR) ≈ (1 + SR²/2) / n
            variance_estimate = (1 + sharpe**2 / 2) / n_obs
        
        # PSR: probability that true Sharpe > benchmark
        psr = stats.norm.cdf((sharpe - self.benchmark_sharpe) / np.sqrt(variance_estimate))
        
        # Deflated Sharpe Ratio adjusts for multiple testing
        if n_trials > 1:
            # Expected maximum Sharpe under null hypothesis
            expected_max_sharpe = self.benchmark_sharpe + np.sqrt(variance_estimate) * self._expected_maximum_of_normals(n_trials)
            
            # DSR is probability that observed Sharpe > expected maximum under null
            dsr = stats.norm.cdf((sharpe - expected_max_sharpe) / np.sqrt(variance_estimate))
        else:
            dsr = psr
        
        # P-value for hypothesis test
        z_score = (sharpe - self.benchmark_sharpe) / np.sqrt(variance_estimate)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
        
        return {
            'dsr': dsr,
            'psr': psr, 
            'sharpe': sharpe,
            'variance_estimate': variance_estimate,
            'p_value': p_value,
            'n_trials': n_trials,
            'n_observations': n_obs
        }
    
    def _expected_maximum_of_normals(self, n: int) -> float:
        """Expected value of maximum of n independent standard normal variables"""
        if n <= 1:
            return 0.0
        
        # Asymptotic approximation for large n
        if n > 100:
            return np.sqrt(2 * np.log(n))
        
        # Use numerical integration for smaller n
        from scipy.special import erf
        
        # Monte Carlo approximation for moderate n
        if n <= 10:
            # Exact calculation for small n is expensive, use approximation
            return np.sqrt(2 * np.log(n)) - (np.log(np.log(n)) + np.log(4*np.pi)) / (2*np.sqrt(2*np.log(n)))
        
        # Better approximation for moderate n
        gamma = 0.5772156649  # Euler-Mascheroni constant
        return np.sqrt(2 * np.log(n)) - (np.log(np.log(n)) + np.log(4*np.pi) - 2*gamma) / (2*np.sqrt(2*np.log(n)))

class CombinatorlialPurgedCV:
    """
    Enhanced Combinatorial Purged Cross-Validation
    Builds on the basic CPCV with additional sophistication for financial time series
    """
    
    def __init__(self, n_splits: int = 6, embargo_pct: float = 0.02, 
                 purge_pct: float = 0.02, test_pct: float = 0.2):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct  
        self.test_pct = test_pct
        
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
              groups: Optional[pd.Series] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits with purging and embargo
        
        Args:
            X: Feature matrix with datetime index or 'ts' column
            y: Target series (optional)
            groups: Grouping variable (optional)
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        if 'ts' in X.columns:
            timestamps = pd.to_datetime(X['ts'])
        elif isinstance(X.index, pd.DatetimeIndex):
            timestamps = X.index
        else:
            raise ValueError("X must have datetime index or 'ts' column")
        
        # Get unique dates and sort
        unique_dates = sorted(timestamps.unique())
        n_dates = len(unique_dates)
        
        # Calculate embargo and purge windows
        embargo_days = max(1, int(n_dates * self.embargo_pct))
        purge_days = max(1, int(n_dates * self.purge_pct))
        test_days = max(1, int(n_dates * self.test_pct))
        
        splits = []
        
        # Create splits ensuring no overlap and proper purging
        for i in range(self.n_splits):
            # Test period placement (evenly spaced)
            test_start_idx = i * (n_dates - test_days) // (self.n_splits - 1) if self.n_splits > 1 else 0
            test_end_idx = min(test_start_idx + test_days, n_dates)
            
            # Purge and embargo around test period
            purge_start = max(0, test_start_idx - purge_days)
            embargo_end = min(n_dates, test_end_idx + embargo_days)
            
            # Training indices (excluding purged/embargoed periods)
            train_dates = (
                unique_dates[:purge_start] + 
                unique_dates[embargo_end:]
            )
            test_dates = unique_dates[test_start_idx:test_end_idx]
            
            # Convert back to indices
            train_idx = timestamps.isin(train_dates)
            test_idx = timestamps.isin(test_dates)
            
            if train_idx.sum() > 0 and test_idx.sum() > 0:
                splits.append((np.where(train_idx)[0], np.where(test_idx)[0]))
        
        return splits
    
    def validate_splits(self, splits: List[Tuple[np.ndarray, np.ndarray]], 
                       timestamps: pd.Series) -> Dict[str, Any]:
        """Validate that splits properly implement purging and embargo"""
        validation_results = {
            'valid': True,
            'warnings': [],
            'stats': {}
        }
        
        overlaps = 0
        train_sizes = []
        test_sizes = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            train_sizes.append(len(train_idx))
            test_sizes.append(len(test_idx))
            
            # Check for overlap
            if set(train_idx).intersection(set(test_idx)):
                overlaps += 1
                validation_results['warnings'].append(f"Split {i}: Train/test overlap detected")
            
            # Check temporal ordering (train should not contain future of test)
            if len(train_idx) > 0 and len(test_idx) > 0:
                max_train_date = timestamps.iloc[train_idx].max()
                min_test_date = timestamps.iloc[test_idx].min()
                
                if max_train_date > min_test_date:
                    validation_results['warnings'].append(f"Split {i}: Train data contains future of test data")
        
        validation_results['stats'] = {
            'n_splits': len(splits),
            'overlaps': overlaps,
            'avg_train_size': np.mean(train_sizes),
            'avg_test_size': np.mean(test_sizes),
            'train_size_std': np.std(train_sizes),
            'test_size_std': np.std(test_sizes)
        }
        
        if overlaps > 0:
            validation_results['valid'] = False
        
        return validation_results

class WalkForwardAnalysis:
    """
    Walk-Forward Analysis with sophisticated validation metrics
    """
    
    def __init__(self, train_window_months: int = 36, test_window_months: int = 6,
                 step_months: int = 3, min_observations: int = 252):
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months  
        self.step_months = step_months
        self.min_observations = min_observations
        
    def generate_windows(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[Dict[str, pd.Timestamp]]:
        """Generate walk-forward windows"""
        windows = []
        current_date = start_date
        
        while current_date + pd.DateOffset(months=self.train_window_months + self.test_window_months) <= end_date:
            train_start = current_date
            train_end = current_date + pd.DateOffset(months=self.train_window_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_window_months)
            
            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })
            
            current_date += pd.DateOffset(months=self.step_months)
        
        return windows
    
    def run_analysis(self, data: pd.DataFrame, model_func, target_col: str = 'target') -> Dict[str, Any]:
        """
        Run walk-forward analysis
        
        Args:
            data: DataFrame with datetime index and features
            model_func: Function that takes (X_train, y_train, X_test) and returns predictions
            target_col: Name of target column
            
        Returns:
            Analysis results
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have datetime index")
        
        windows = self.generate_windows(data.index.min(), data.index.max())
        results = []
        
        for i, window in enumerate(windows):
            # Split data
            train_mask = (data.index >= window['train_start']) & (data.index < window['train_end'])
            test_mask = (data.index >= window['test_start']) & (data.index < window['test_end'])
            
            train_data = data[train_mask]
            test_data = data[test_mask]
            
            if len(train_data) < self.min_observations or len(test_data) == 0:
                continue
            
            # Prepare features and targets
            feature_cols = [col for col in data.columns if col != target_col]
            X_train = train_data[feature_cols]
            y_train = train_data[target_col]
            X_test = test_data[feature_cols]
            y_test = test_data[target_col]
            
            try:
                # Generate predictions
                predictions = model_func(X_train, y_train, X_test)
                
                # Calculate metrics
                window_results = self._calculate_window_metrics(
                    y_test, predictions, window, i
                )
                results.append(window_results)
                
            except Exception as e:
                log.error(f"Error in window {i}: {e}")
                continue
        
        # Aggregate results
        return self._aggregate_results(results)
    
    def _calculate_window_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                                window: Dict, window_idx: int) -> Dict[str, Any]:
        """Calculate metrics for a single window"""
        returns = y_true * y_pred  # Assuming y_pred are position signals
        
        metrics = {
            'window_idx': window_idx,
            'train_start': window['train_start'],
            'train_end': window['train_end'],
            'test_start': window['test_start'],
            'test_end': window['test_end'],
            'n_observations': len(y_true),
            'ic': np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0,
            'returns_mean': returns.mean(),
            'returns_std': returns.std(),
            'sharpe': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0,
            'max_drawdown': self._calculate_max_drawdown(returns.cumsum()),
            'hit_rate': (returns > 0).mean()
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()
    
    def _aggregate_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across windows"""
        if not results:
            return {}
        
        df = pd.DataFrame(results)
        
        # Overall performance
        all_returns = []
        for result in results:
            # This is simplified - in practice would need actual returns
            all_returns.extend([result['returns_mean']] * result['n_observations'])
        
        all_returns = pd.Series(all_returns)
        
        aggregated = {
            'n_windows': len(results),
            'avg_ic': df['ic'].mean(),
            'ic_std': df['ic'].std(),
            'ic_ir': df['ic'].mean() / df['ic'].std() if df['ic'].std() > 0 else 0.0,
            'avg_sharpe': df['sharpe'].mean(),
            'sharpe_std': df['sharpe'].std(),
            'avg_hit_rate': df['hit_rate'].mean(),
            'overall_sharpe': all_returns.mean() / all_returns.std() * np.sqrt(252) if all_returns.std() > 0 else 0.0,
            'max_drawdown': df['max_drawdown'].min(),
            'windows': results
        }
        
        return aggregated