//! Linear models for regression and classification.
//!
//! This module provides implementations of linear models including:
//! - `LinearRegression`: Ordinary least squares linear regression
//! - `LogisticRegression`: Logistic regression for binary classification
//!
//! # Examples
//!
//! ## Linear Regression
//! ```rust
//! use nametbd::{LinearRegression, Matrix, Vector};
//! use ndarray::array;
//!
//! let x = array![[1.0], [2.0], [3.0]];
//! let y = array![2.0, 4.0, 6.0];
//!
//! let mut model = LinearRegression::new();
//! model.fit(&x, &y).unwrap();
//! let predictions = model.predict(&x).unwrap();
//! ```
//!
//! ## Logistic Regression
//! ```rust
//! use nametbd::{LogisticRegression, Matrix, Vector};
//! use ndarray::array;
//!
//! let x = array![[1.0], [2.0], [3.0], [4.0]];
//! let y = array![0.0, 0.0, 1.0, 1.0];
//!
//! let mut model = LogisticRegression::new();
//! model.fit(&x, &y).unwrap();
//! let predictions = model.predict(&x).unwrap();
//! let probabilities = model.predict_proba(&x).unwrap();
//! ```

mod linear_regression;
mod logistic_regression;
mod ridge;
mod lasso;
mod elastic_net;
mod svm;

pub use linear_regression::LinearRegression;
pub use logistic_regression::LogisticRegression;
pub use ridge::Ridge;
pub use lasso::Lasso;
pub use elastic_net::ElasticNet;
pub use svm::{SVC, Kernel};