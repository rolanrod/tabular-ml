//! Clustering algorithms for unsupervised learning.
//!
//! This module provides implementations of clustering algorithms including:
//! - `KMeans`: Partitional clustering using centroids
//! - `DBSCAN`: Density-based clustering for arbitrary shaped clusters
//!
//! # Examples
//!
//! ## K-Means Clustering
//! ```rust
//! use nametbd::{KMeans, Matrix};
//! use ndarray::array;
//!
//! let x = array![
//!     [1.0, 1.0],
//!     [1.5, 2.0],
//!     [3.0, 4.0],
//!     [5.0, 7.0],
//!     [3.5, 5.0],
//!     [4.5, 5.0]
//! ];
//!
//! let mut kmeans = KMeans::new(2).max_iter(100);
//! let labels = kmeans.fit_predict(&x).unwrap();
//! 
//! // Get cluster centers
//! let centers = kmeans.cluster_centers.as_ref().unwrap();
//! println!("Cluster centers: {:?}", centers);
//! 
//! // Get within-cluster sum of squares
//! let inertia = kmeans.inertia.unwrap();
//! println!("Inertia: {:.4}", inertia);
//! ```
//!
//! ## DBSCAN Clustering
//! ```rust
//! use nametbd::{DBSCAN, Matrix};
//! use ndarray::array;
//!
//! let x = array![
//!     [1.0, 1.0],
//!     [1.2, 1.1],
//!     [1.1, 1.2],
//!     [8.0, 8.0],
//!     [8.1, 8.1],
//!     [8.2, 7.9],
//!     [15.0, 1.0] // Outlier
//! ];
//!
//! let mut dbscan = DBSCAN::new(1.0, 2); // eps=1.0, min_samples=2
//! let labels = dbscan.fit_predict(&x).unwrap();
//! 
//! // Get number of clusters found
//! let n_clusters = dbscan.get_n_clusters().unwrap();
//! println!("Number of clusters: {}", n_clusters);
//! 
//! // Get number of noise points
//! let n_noise = dbscan.get_n_noise_points().unwrap();
//! println!("Number of noise points: {}", n_noise);
//! ```

mod kmeans;
mod dbscan;

pub use kmeans::KMeans;
pub use dbscan::DBSCAN;