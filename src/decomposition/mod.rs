//! Dimensionality reduction and matrix decomposition algorithms.
//!
//! This module provides implementations of dimensionality reduction techniques including:
//! - `PCA`: Principal Component Analysis for unsupervised dimensionality reduction
//! - `LDA`: Linear Discriminant Analysis for supervised dimensionality reduction
//! - `TruncatedSVD`: Truncated Singular Value Decomposition
//!
//! # Examples
//!
//! ## Principal Component Analysis (PCA)
//! ```rust
//! use nametbd::{PCA, Matrix};
//! use ndarray::array;
//!
//! let x = array![
//!     [1.0, 2.0, 3.0],
//!     [4.0, 5.0, 6.0],
//!     [7.0, 8.0, 9.0]
//! ];
//!
//! let mut pca = PCA::new().n_components(2);
//! let transformed = pca.fit_transform(&x).unwrap();
//! 
//! // Get explained variance ratio
//! let explained_var = pca.explained_variance_ratio.as_ref().unwrap();
//! println!("Explained variance ratio: {:?}", explained_var);
//! ```
//!
//! ## Linear Discriminant Analysis (LDA)
//! ```rust
//! use nametbd::{LDA, Matrix, Vector};
//! use ndarray::array;
//!
//! let x = array![
//!     [1.0, 2.0],
//!     [2.0, 3.0],
//!     [8.0, 9.0],
//!     [9.0, 10.0]
//! ];
//! let y = array![0.0, 0.0, 1.0, 1.0];
//!
//! let mut lda = LDA::new();
//! let transformed = lda.fit_transform(&x, &y).unwrap();
//! 
//! // Make predictions on new data
//! let predictions = lda.predict(&x).unwrap();
//! ```
//!
//! ## Truncated SVD
//! ```rust
//! use nametbd::{TruncatedSVD, Matrix};
//! use ndarray::array;
//!
//! let x = array![
//!     [1.0, 2.0, 3.0, 4.0],
//!     [5.0, 6.0, 7.0, 8.0],
//!     [9.0, 10.0, 11.0, 12.0]
//! ];
//!
//! let mut svd = TruncatedSVD::new(2);
//! let transformed = svd.fit_transform(&x).unwrap();
//! 
//! // Reconstruct original data (with dimensionality reduction)
//! let reconstructed = svd.inverse_transform(&transformed).unwrap();
//! ```

mod pca;
mod lda;
mod truncated_svd;

pub use pca::PCA;
pub use lda::LDA;
pub use truncated_svd::TruncatedSVD;
