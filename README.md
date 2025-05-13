# Human Activity Recognition using Matrix Profile and Machine Learning

## Overview

This project implements a human activity recognition system using motion time series data. The system applies Matrix Profile extraction and leverages machine learning models—Support Vector Machines (SVM) and K-Nearest Neighbors (KNN)—to classify 19 different types of physical activities. The project is a part of a Knowledge Engineering course and focuses on applying time-series similarity techniques for pattern recognition.

## Dataset

* **Source:** [UCI Daily and Sports Activities Dataset](https://archive.ics.uci.edu/ml/datasets/Daily+and+Sports+Activities)
* **Content:** Multivariate time series from body-worn sensors across multiple physical activities (e.g., walking, running, squatting).
* **Downloaded with:**

  ```bash
  !wget https://archive.ics.uci.edu/static/public/256/daily+and+sports+activities.zip -O daily_sports_activities.zip
  !unzip daily_sports_activities.zip -d daily_sports_activities
  ```

## Methodology

### Step 1: Matrix Profile Feature Extraction

* Extracted the `torso_x` coordinate from raw files.
* Computed Matrix Profiles using Euclidean distance for each subsequence (length `m=50`).
* Generated a 76-dimensional feature vector for each sample.

### Step 2: Data Preprocessing

* Standardized features using `StandardScaler`.
* Created training and test splits (80/20 ratio).
* Exported standardized datasets to CSV files for reuse.

### Step 3: Classification Models

#### Support Vector Machine (SVM)

* Performed Grid Search with `GridSearchCV` for hyperparameter tuning.
* Best Parameters:

  ```json
  {"C": 100, "gamma": 0.1, "kernel": "rbf", "class_weight": "balanced"}
  ```
* **Accuracy:** 44.30%

#### K-Nearest Neighbors (KNN)

* Used `n_neighbors=5`
* **Accuracy:** 46.00%

## Results

| Model | Accuracy | Highlights                                                    |
| ----- | -------- | ------------------------------------------------------------- |
| SVM   | 44.30%   | Balanced class weighting improved minority class performance. |
| KNN   | 46.00%   | Achieved slightly better generalization on the test set.      |

* Both models show promising potential given the complex nature of time-series similarity-based features.
* Confusion matrix and per-class metrics suggest performance variance across activity types.

## Sample Output

```bash
Feature Matrix Size: 1140
Length of the first feature vector: 76
Accuracy: 0.46 (KNN)
Accuracy: 0.443 (SVM)
```

## Technologies Used

* Python 3
* NumPy, Pandas, Scikit-learn
* Matrix Profile computation (custom implementation)

## Future Work

* Incorporate other body joints or multiple axes for richer feature representations.
* Use STUMPY or MASS algorithms for optimized matrix profile computation.
* Try ensemble learning methods (e.g., Random Forest, Gradient Boosting).

## Author

**Omer Efe Kaymaz**

## License

This project is licensed under the MIT License.
