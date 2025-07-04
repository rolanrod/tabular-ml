use crate::{Matrix, Vector};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct KMeans {
    pub cluster_centers: Option<Matrix>,
    pub labels: Option<Vector>,
    pub inertia: Option<f64>,
    n_clusters: usize,
    max_iter: usize,
    tolerance: f64,
    random_state: Option<u64>,
    init: String,
}

impl KMeans {
    pub fn new(n_clusters: usize) -> Self {
        if n_clusters == 0 {
            panic!("n_clusters must be > 0, got {}", n_clusters);
        }
        
        Self {
            cluster_centers: None,
            labels: None,
            inertia: None,
            n_clusters,
            max_iter: 300,
            tolerance: 1e-4,
            random_state: None,
            init: "k-means++".to_string(),
        }
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    pub fn init(mut self, init: &str) -> Self {
        match init {
            "k-means++" | "random" => {
                self.init = init.to_string();
            }
            _ => panic!("Invalid init method: {}. Must be 'k-means++' or 'random'", init),
        }
        self
    }

    pub fn fit(&mut self, x: &Matrix) -> Result<(), String> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err("Input matrix must have at least one sample and one feature".to_string());
        }

        if x.nrows() < self.n_clusters {
            return Err(format!(
                "n_samples={} should be >= n_clusters={}",
                x.nrows(), self.n_clusters
            ));
        }

        // Initialize centroids
        let mut centroids = self.initialize_centroids(x)?;
        let mut labels = Vector::zeros(x.nrows());
        
        for iteration in 0..self.max_iter {
            let old_centroids = centroids.clone();
            
            // Assign points to nearest centroid
            for i in 0..x.nrows() {
                let mut min_distance = f64::INFINITY;
                let mut closest_cluster = 0;
                
                for k in 0..self.n_clusters {
                    let distance = self.euclidean_distance(&x.row(i), &centroids.row(k));
                    if distance < min_distance {
                        min_distance = distance;
                        closest_cluster = k;
                    }
                }
                
                labels[i] = closest_cluster as f64;
            }
            
            // Update centroids
            for k in 0..self.n_clusters {
                let cluster_points: Vec<usize> = labels.iter()
                    .enumerate()
                    .filter(|(_, &label)| label == k as f64)
                    .map(|(i, _)| i)
                    .collect();
                
                if !cluster_points.is_empty() {
                    for j in 0..x.ncols() {
                        let sum: f64 = cluster_points.iter()
                            .map(|&point_idx| x[[point_idx, j]])
                            .sum();
                        centroids[[k, j]] = sum / cluster_points.len() as f64;
                    }
                }
            }
            
            // Check for convergence
            let centroid_shift = self.max_centroid_shift(&old_centroids, &centroids);
            if centroid_shift < self.tolerance {
                break;
            }
        }
        
        // Calculate inertia (within-cluster sum of squares)
        let mut inertia = 0.0;
        for i in 0..x.nrows() {
            let cluster_idx = labels[i] as usize;
            let distance = self.euclidean_distance(&x.row(i), &centroids.row(cluster_idx));
            inertia += distance * distance;
        }
        
        self.cluster_centers = Some(centroids);
        self.labels = Some(labels);
        self.inertia = Some(inertia);
        
        Ok(())
    }

    pub fn predict(&self, x: &Matrix) -> Result<Vector, String> {
        let centroids = self.cluster_centers.as_ref()
            .ok_or("KMeans not fitted. Call fit() first.")?;

        if x.ncols() != centroids.ncols() {
            return Err(format!(
                "Number of features in X ({}) doesn't match training data ({})",
                x.ncols(), centroids.ncols()
            ));
        }

        let mut labels = Vector::zeros(x.nrows());
        
        for i in 0..x.nrows() {
            let mut min_distance = f64::INFINITY;
            let mut closest_cluster = 0;
            
            for k in 0..self.n_clusters {
                let distance = self.euclidean_distance(&x.row(i), &centroids.row(k));
                if distance < min_distance {
                    min_distance = distance;
                    closest_cluster = k;
                }
            }
            
            labels[i] = closest_cluster as f64;
        }
        
        Ok(labels)
    }

    pub fn fit_predict(&mut self, x: &Matrix) -> Result<Vector, String> {
        self.fit(x)?;
        Ok(self.labels.as_ref().unwrap().clone())
    }

    pub fn transform(&self, x: &Matrix) -> Result<Matrix, String> {
        let centroids = self.cluster_centers.as_ref()
            .ok_or("KMeans not fitted. Call fit() first.")?;

        if x.ncols() != centroids.ncols() {
            return Err(format!(
                "Number of features in X ({}) doesn't match training data ({})",
                x.ncols(), centroids.ncols()
            ));
        }

        let mut distances = Matrix::zeros((x.nrows(), self.n_clusters));
        
        for i in 0..x.nrows() {
            for k in 0..self.n_clusters {
                distances[[i, k]] = self.euclidean_distance(&x.row(i), &centroids.row(k));
            }
        }
        
        Ok(distances)
    }

    fn initialize_centroids(&self, x: &Matrix) -> Result<Matrix, String> {
        let mut centroids = Matrix::zeros((self.n_clusters, x.ncols()));
        
        match self.init.as_str() {
            "random" => {
                // Simple random initialization
                for k in 0..self.n_clusters {
                    let random_idx = (k * 17 + 42) % x.nrows(); // Simple pseudo-random
                    centroids.row_mut(k).assign(&x.row(random_idx));
                }
            }
            "k-means++" => {
                // K-means++ initialization
                // First centroid: random point
                let first_idx = self.random_state.unwrap_or(42) as usize % x.nrows();
                centroids.row_mut(0).assign(&x.row(first_idx));
                
                for k in 1..self.n_clusters {
                    let mut distances = Vector::zeros(x.nrows());
                    
                    // Calculate distance to nearest existing centroid
                    for i in 0..x.nrows() {
                        let mut min_dist = f64::INFINITY;
                        for j in 0..k {
                            let dist = self.euclidean_distance(&x.row(i), &centroids.row(j));
                            min_dist = min_dist.min(dist);
                        }
                        distances[i] = min_dist * min_dist; // Squared distance
                    }
                    
                    // Choose next centroid proportional to squared distance
                    let total_dist: f64 = distances.sum();
                    if total_dist > 0.0 {
                        let mut cumulative = 0.0;
                        let target = (self.random_state.unwrap_or(42 + k as u64) as f64 / u64::MAX as f64) * total_dist;
                        
                        for i in 0..x.nrows() {
                            cumulative += distances[i];
                            if cumulative >= target {
                                centroids.row_mut(k).assign(&x.row(i));
                                break;
                            }
                        }
                    } else {
                        // Fallback: use a different point
                        let fallback_idx = (k * 23 + 17) % x.nrows();
                        centroids.row_mut(k).assign(&x.row(fallback_idx));
                    }
                }
            }
            _ => return Err(format!("Unknown initialization method: {}", self.init)),
        }
        
        Ok(centroids)
    }

    fn euclidean_distance(&self, a: &ndarray::ArrayView1<f64>, b: &ndarray::ArrayView1<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    fn max_centroid_shift(&self, old_centroids: &Matrix, new_centroids: &Matrix) -> f64 {
        let mut max_shift = 0.0;
        
        for k in 0..self.n_clusters {
            let shift = self.euclidean_distance(&old_centroids.row(k), &new_centroids.row(k));
            max_shift = max_shift.max(shift);
        }
        
        max_shift
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_kmeans_basic() {
        let x = array![
            [1.0, 1.0],
            [1.5, 2.0],
            [3.0, 4.0],
            [5.0, 7.0],
            [3.5, 5.0],
            [4.5, 5.0],
            [3.5, 4.5]
        ];

        let mut kmeans = KMeans::new(2);
        let labels = kmeans.fit_predict(&x).unwrap();
        
        assert_eq!(labels.len(), x.nrows());
        assert!(kmeans.cluster_centers.is_some());
        assert!(kmeans.inertia.is_some());
        
        // Check that we have exactly 2 unique labels
        let unique_labels: std::collections::HashSet<i32> = labels.iter()
            .map(|&x| x as i32)
            .collect();
        assert_eq!(unique_labels.len(), 2);
    }

    #[test]
    fn test_kmeans_predict() {
        let x_train = array![
            [0.0, 0.0],
            [1.0, 1.0],
            [10.0, 10.0],
            [11.0, 11.0]
        ];
        
        let x_test = array![
            [0.5, 0.5],
            [10.5, 10.5]
        ];

        let mut kmeans = KMeans::new(2);
        kmeans.fit(&x_train).unwrap();
        
        let labels = kmeans.predict(&x_test).unwrap();
        assert_eq!(labels.len(), 2);
    }

    #[test]
    fn test_kmeans_transform() {
        let x = array![
            [0.0, 0.0],
            [1.0, 1.0],
            [10.0, 10.0]
        ];

        let mut kmeans = KMeans::new(2);
        kmeans.fit(&x).unwrap();
        
        let distances = kmeans.transform(&x).unwrap();
        assert_eq!(distances.shape(), &[3, 2]); // 3 samples, 2 clusters
        
        // All distances should be non-negative
        for distance in distances.iter() {
            assert!(*distance >= 0.0);
        }
    }

    #[test]
    fn test_kmeans_invalid_clusters() {
        std::panic::catch_unwind(|| {
            KMeans::new(0);
        }).expect_err("Should panic on zero clusters");
    }

    #[test]
    fn test_kmeans_insufficient_samples() {
        let x = array![[1.0, 2.0]]; // Only 1 sample
        let mut kmeans = KMeans::new(2); // But want 2 clusters
        
        assert!(kmeans.fit(&x).is_err());
    }

    #[test]
    fn test_kmeans_predict_without_fit() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let kmeans = KMeans::new(2);
        
        assert!(kmeans.predict(&x).is_err());
    }

    #[test]
    fn test_kmeans_dimension_mismatch() {
        let x_train = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let x_test = array![[1.0, 2.0], [3.0, 4.0]]; // Different dimensions
        
        let mut kmeans = KMeans::new(2);
        kmeans.fit(&x_train).unwrap();
        
        assert!(kmeans.predict(&x_test).is_err());
    }
}