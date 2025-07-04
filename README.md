# Tabular ML

**Tabular-ML** is a Rust library I built primarily to practice using the Rust programming language and familiarizing myself with its package system. I chose a project around tabular ML because it's what I know best! The project implements core tabular machine learning algorithms, designed for learning, experimentation, and practical use on structured data. It provides a familiar API for users coming from Python's scikit-learn, but leverages Rust's performance and safety.

## Features

- **Linear Models**
  - Linear Regression (Ordinary Least Squares)
  - Ridge Regression (L2 regularization)
  - Lasso Regression (L1 regularization)
  - Elastic Net (L1 + L2 regularization)
  - Logistic Regression (binary classification)
  - Support Vector Machine (SVM) for binary classification (multiple kernels)

- **Clustering**
  - K-Means
  - DBSCAN

- **Dimensionality Reduction**
  - Principal Component Analysis (PCA)
  - Linear Discriminant Analysis (LDA)
  - Truncated Singular Value Decomposition (TruncatedSVD)

- **Preprocessing**
  - StandardScaler (feature standardization)
  - Dataset utilities (train/test split, etc.)

- **Metrics**
  - Regression and classification metrics (R², MSE, accuracy, precision, recall, F1, confusion matrix, etc.)

## Getting Started

Add to your `Cargo.toml`:

```toml
[dependencies]
nametbd = { path = "path/to/this/repo" }
ndarray = "0.16"
```

## Example Usage

### Linear Regression

```rust
use nametbd::{LinearRegression, Matrix, Vector};
use ndarray::array;

fn main() -> Result<(), String> {
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![5.1, 6.9, 9.2, 11.1, 12.8];

    let mut model = LinearRegression::new();
    model.fit(&x, &y)?;
    let predictions = model.predict(&x)?;

    println!("Predictions: {:?}", predictions);
    Ok(())
}
```

### K-Means Clustering

```rust
use nametbd::{KMeans, Matrix};
use ndarray::array;

let x = array![
    [1.0, 1.0],
    [1.5, 2.0],
    [3.0, 4.0],
    [5.0, 7.0],
    [3.5, 5.0],
    [4.5, 5.0]
];

let mut kmeans = KMeans::new(2).max_iter(100);
let labels = kmeans.fit_predict(&x).unwrap();
println!("Cluster labels: {:?}", labels);
```

### Principal Component Analysis (PCA)

```rust
use nametbd::{PCA, Matrix};
use ndarray::array;

let x = array![
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
];

let mut pca = PCA::new().n_components(2);
let transformed = pca.fit_transform(&x).unwrap();
println!("PCA output: {:?}", transformed);
```

## Examples

See the [`examples/`](examples/) directory for:
- Linear regression, logistic regression, and regularized regression
- Clustering (KMeans, DBSCAN)
- Dimensionality reduction (PCA, LDA, TruncatedSVD)
- SVM classification

Run an example with:

```sh
cargo run --example simple_example
```

## Design Goals

- **Educational**: Clear, readable implementations of core algorithms.
- **Extensible**: Modular design for adding new models and utilities.
- **Safe and Fast**: Leverages Rust's safety and performance.
- **Familiar API**: Inspired by scikit-learn for easy adoption.

## Roadmap

Planned features include:
- Tree-based models (Decision Trees, Random Forests, Gradient Boosting)
- Naive Bayes classifiers
- Ensemble methods (Bagging, AdaBoost, Voting)
- Model selection and validation (cross-validation, grid search)
