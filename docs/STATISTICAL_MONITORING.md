# Statistical Monitoring Implementation

## Overview

AsekioML now includes real statistical calculations for monitoring model and data drift in production environments. This document describes the implementation of PSI (Population Stability Index) and Wasserstein distance calculations.

## Features Implemented

### 1. Population Stability Index (PSI)

**Purpose**: Measures the stability of a population by comparing the distribution of variables across different time periods.

**Implementation**: 
- Uses decile-based binning (10 bins) for consistent comparison
- Bins are defined using reference data percentiles
- Formula: `PSI = Σ((current_pct - reference_pct) * ln(current_pct / reference_pct))`
- Handles multiple features by averaging PSI across all features
- Includes epsilon (1e-10) to prevent log(0) errors

**Interpretation**:
- PSI < 0.1: No significant population change
- 0.1 ≤ PSI < 0.2: Minor population change
- PSI ≥ 0.2: Major population change (action required)

### 2. Wasserstein Distance (Earth Mover's Distance)

**Purpose**: Measures the distance between two probability distributions, providing insight into how much "work" is needed to transform one distribution into another.

**Implementation**:
- Calculates 1-Wasserstein distance using quantile-based approximation
- Uses 100 quantiles for accurate approximation
- Formula: `W1(P,Q) = ∫|F_P^{-1}(u) - F_Q^{-1}(u)| du` from 0 to 1
- Handles multivariate data by averaging distance across features
- Sorts data for quantile calculation

**Interpretation**:
- Lower values indicate more similar distributions
- Higher values indicate significant distribution shift
- Units are in the same scale as the original data

### 3. Kolmogorov-Smirnov (KS) Test

**Purpose**: Measures the maximum difference between the cumulative distribution functions (CDFs) of two datasets, providing a non-parametric test for distribution similarity.

**Implementation**:
- Calculates KS statistic: `max|F_ref(x) - F_cur(x)|` where F(x) is the CDF
- Handles multivariate data by taking maximum KS across features
- Uses efficient merge-sort approach for CDF calculation
- Returns values in range [0, 1]

**Interpretation**:
- 0 indicates identical distributions
- Values closer to 1 indicate more significant differences
- Typical significance thresholds: 0.05-0.1 depending on sample size
- More sensitive to shape changes than location/scale shifts

### 4. Chi-square Test

**Purpose**: Tests for independence between categorical distributions, ideal for detecting drift in discrete or binned continuous data.

**Implementation**:
- Discretizes continuous data into 10 bins for categorical analysis
- Calculates Chi-square statistic: `Σ((observed - expected)² / expected)`
- Expected frequencies based on reference data proportions
- Handles multivariate data by averaging across features
- Includes epsilon values to prevent division by zero

**Interpretation**:
- Higher values indicate more significant differences between distributions
- Critical values depend on degrees of freedom and significance level
- Particularly effective for categorical data and frequency-based drift
- Complements other tests by focusing on categorical patterns

### 5. Model Monitoring Integration

**ModelMonitor Class**:
- Tracks reference data as baseline
- Configurable thresholds for drift detection
- Real-time drift detection using PSI, Wasserstein, KS, and Chi-square metrics
- Thread-safe monitoring capabilities
- Comprehensive statistical test suite for different data types

## Usage Example

```cpp
#include "production_features.hpp"

// Create baseline dataset
Dataset reference_data = load_reference_data();
ModelMonitor monitor("production_model", reference_data, 0.15, 0.1);

// Monitor new data
Dataset new_data = get_latest_batch();
auto metrics = monitor.detect_data_drift(new_data);

if (metrics.drift_detected) {
    std::cout << "Drift detected!" << std::endl;
    std::cout << "  PSI: " << metrics.psi_score << std::endl;
    std::cout << "  Wasserstein: " << metrics.wasserstein_distance << std::endl;
    std::cout << "  KS Statistic: " << metrics.ks_statistic << std::endl;
    std::cout << "  Chi-square: " << metrics.chi_square_statistic << std::endl;
    // Take action: retrain model, investigate data source, etc.
}
```

## Technical Details

### Performance Characteristics
- **PSI Calculation**: O(n log n) due to sorting for binning
- **Wasserstein Distance**: O(n log n) due to sorting for quantiles
- **KS Statistic**: O(n log n) due to sorting and CDF merging
- **Chi-square Test**: O(n log n) due to binning and frequency counting
- **Memory Usage**: Linear in input size, no significant overhead
- **Numerical Stability**: Uses epsilon values to prevent edge cases

### Limitations and Considerations
1. **Binning Strategy**: Fixed 10-bin approach may not suit all data types
2. **Multivariate Handling**: Currently averages across features (future: proper multivariate metrics)
3. **Sample Size**: Performance is better with larger samples (>100 recommended)
4. **Data Types**: Optimized for continuous numerical data
5. **Statistical Significance**: No automatic p-value calculations (thresholds are user-defined)

### Future Enhancements
- Adaptive binning strategies
- True multivariate Wasserstein distance
- P-value calculations for statistical significance testing
- Categorical data support with proper Chi-square degrees of freedom
- Time-series specific drift detection
- Bootstrap confidence intervals for test statistics

## Validation

The implementations have been validated through:
1. **Synthetic Data Tests**: Verified correct behavior with known distributions
2. **Edge Case Testing**: Tested with small datasets and identical data
3. **Production Simulation**: Multi-week drift simulation showing gradual detection
4. **Benchmark Comparison**: Results align with expected statistical behavior

## Integration with AsekioML

These statistical monitoring capabilities integrate seamlessly with AsekioML's production features:
- **Thread Safety**: All calculations are thread-safe
- **Performance Monitoring**: Integrated with profiling system
- **Error Handling**: Robust error checking and graceful fallbacks
- **Zero Dependencies**: Pure C++ implementation with no external libraries

This addresses key Python ML framework pain points:
- **Transparency**: Full visibility into calculation methods
- **Performance**: Native C++ speed vs Python overhead
- **Dependencies**: No external statistical libraries required
- **Debugging**: Direct access to intermediate calculations
