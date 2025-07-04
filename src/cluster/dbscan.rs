use crate::{Matrix, Vector};
use std::collections::{HashSet, VecDeque};

#[derive(Clone, Debug)]
pub struct DBSCAN {
    pub labels: Option<Vector>,
    pub core_sample_indices: Option<Vec<usize>>,
    eps: f64,
    min_samples: usize,
    metric: String,
}

impl DBSCAN {
    pub fn new(eps: f64, min_samples: usize) -> Self {
        if eps <= 0.0 {
            panic!("eps must be > 0, got {}", eps);
        }
        if min_samples == 0 {
            panic!("min_samples must be > 0, got {}", min_samples);
        }
        
        Self {
            labels: None,
            core_sample_indices: None,
            eps,
            min_samples,
            metric: "euclidean".to_string(),
        }
    }

    pub fn metric(mut self, metric: &str) -> Self {
        match metric {
            "euclidean" | "manhattan" => {
                self.metric = metric.to_string();
            }
            _ => panic!("Invalid metric: {}. Must be 'euclidean' or 'manhattan'", metric),
        }
        self
    }

    pub fn fit(&mut self, x: &Matrix) -> Result<(), String> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err("Input matrix must have at least one sample and one feature".to_string());
        }

        let n_samples = x.nrows();
        let mut labels = Vector::from_elem(n_samples, -1.0); // -1 indicates noise/unassigned
        let mut core_samples = Vec::new();
        let mut visited = vec![false; n_samples];
        let mut current_cluster = 0.0;

        // Find all core samples
        for i in 0..n_samples {
            let neighbors = self.region_query(x, i);
            if neighbors.len() >= self.min_samples {
                core_samples.push(i);
            }
        }

        // Process each core sample
        for &core_idx in &core_samples {
            if visited[core_idx] {
                continue;
            }

            visited[core_idx] = true;
            let mut neighbors = self.region_query(x, core_idx);
            labels[core_idx] = current_cluster;

            // Expand cluster using BFS
            let mut queue = VecDeque::new();
            queue.extend(neighbors.iter());

            while let Some(&neighbor_idx) = queue.pop_front() {
                if !visited[neighbor_idx] {
                    visited[neighbor_idx] = true;
                    labels[neighbor_idx] = current_cluster;

                    // If neighbor is also a core sample, add its neighbors to queue
                    if core_samples.contains(&neighbor_idx) {
                        let neighbor_neighbors = self.region_query(x, neighbor_idx);
                        for &nn in &neighbor_neighbors {
                            if labels[nn] == -1.0 { // Only add unassigned points
                                queue.push_back(nn);
                            }
                        }
                    }
                } else if labels[neighbor_idx] == -1.0 {
                    // This is a border point
                    labels[neighbor_idx] = current_cluster;
                }
            }

            current_cluster += 1.0;
        }

        self.labels = Some(labels);
        self.core_sample_indices = Some(core_samples);

        Ok(())
    }

    pub fn fit_predict(&mut self, x: &Matrix) -> Result<Vector, String> {
        self.fit(x)?;
        Ok(self.labels.as_ref().unwrap().clone())
    }

    pub fn predict(&self, x: &Matrix) -> Result<Vector, String> {
        // DBSCAN doesn't naturally support prediction on new points
        // This is a simplified approach that assigns new points to nearest cluster
        let labels = self.labels.as_ref()
            .ok_or("DBSCAN not fitted. Call fit() first.")?;
        let core_indices = self.core_sample_indices.as_ref()
            .ok_or("DBSCAN not fitted. Call fit() first.")?;

        // This would require storing the original training data
        // For now, return an error suggesting fit_predict instead
        Err("DBSCAN doesn't support predict() on new data. Use fit_predict() instead.".to_string())
    }

    fn region_query(&self, x: &Matrix, point_idx: usize) -> Vec<usize> {
        let mut neighbors = Vec::new();
        
        for i in 0..x.nrows() {
            let distance = self.compute_distance(&x.row(point_idx), &x.row(i));
            if distance <= self.eps {
                neighbors.push(i);
            }
        }
        
        neighbors
    }

    fn compute_distance(&self, a: &ndarray::ArrayView1<f64>, b: &ndarray::ArrayView1<f64>) -> f64 {
        match self.metric.as_str() {
            "euclidean" => {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y) * (x - y))
                    .sum::<f64>()
                    .sqrt()
            }
            "manhattan" => {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).abs())
                    .sum::<f64>()
            }
            _ => unreachable!(), // Validated in constructor
        }
    }

    pub fn get_n_clusters(&self) -> Option<usize> {
        self.labels.as_ref().map(|labels| {
            let unique_clusters: HashSet<i32> = labels.iter()
                .map(|&x| x as i32)
                .filter(|&x| x >= 0) // Exclude noise points (-1)
                .collect();
            unique_clusters.len()
        })
    }

    pub fn get_n_noise_points(&self) -> Option<usize> {
        self.labels.as_ref().map(|labels| {
            labels.iter()
                .filter(|&&x| x == -1.0)
                .count()
        })
    }

    pub fn is_core_sample(&self, sample_idx: usize) -> Option<bool> {
        self.core_sample_indices.as_ref().map(|core_indices| {
            core_indices.contains(&sample_idx)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_dbscan_basic() {
        // Create two distinct clusters
        let x = array![
            [1.0, 1.0],
            [1.2, 1.1],
            [1.1, 1.2],
            [8.0, 8.0],
            [8.1, 8.1],
            [8.2, 7.9],
            [15.0, 1.0] // Outlier
        ];

        let mut dbscan = DBSCAN::new(1.0, 2);
        let labels = dbscan.fit_predict(&x).unwrap();
        
        assert_eq!(labels.len(), x.nrows());
        
        // Should find at least 2 clusters
        let n_clusters = dbscan.get_n_clusters().unwrap();
        assert!(n_clusters >= 2);
        
        // Should have some core samples
        assert!(dbscan.core_sample_indices.is_some());
        assert!(!dbscan.core_sample_indices.as_ref().unwrap().is_empty());
    }

    #[test]
    fn test_dbscan_noise_detection() {
        // Sparse points that should be classified as noise
        let x = array![
            [0.0, 0.0],
            [10.0, 10.0],
            [20.0, 20.0],
            [30.0, 30.0]
        ];

        let mut dbscan = DBSCAN::new(1.0, 2); // Need 2 points within distance 1.0
        let labels = dbscan.fit_predict(&x).unwrap();
        
        // All points should be noise (-1) since they're too far apart
        let n_noise = dbscan.get_n_noise_points().unwrap();
        assert_eq!(n_noise, x.nrows());
    }

    #[test]
    fn test_dbscan_single_cluster() {
        // Dense cluster where all points should be in one cluster
        let x = array![
            [1.0, 1.0],
            [1.1, 1.0],
            [1.0, 1.1],
            [1.1, 1.1],
            [1.2, 1.0],
            [1.0, 1.2]
        ];

        let mut dbscan = DBSCAN::new(0.5, 2);
        let labels = dbscan.fit_predict(&x).unwrap();
        
        let n_clusters = dbscan.get_n_clusters().unwrap();
        assert_eq!(n_clusters, 1);
        
        // All points should be in the same cluster (label 0)
        for &label in labels.iter() {
            assert_eq!(label, 0.0);
        }
    }

    #[test]
    fn test_dbscan_manhattan_metric() {
        let x = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ];

        let mut dbscan = DBSCAN::new(1.5, 2).metric("manhattan");
        let labels = dbscan.fit_predict(&x).unwrap();
        
        // With Manhattan distance and eps=1.5, all points should be connected
        let n_clusters = dbscan.get_n_clusters().unwrap();
        assert_eq!(n_clusters, 1);
    }

    #[test]
    fn test_dbscan_invalid_eps() {
        std::panic::catch_unwind(|| {
            DBSCAN::new(-1.0, 2);
        }).expect_err("Should panic on negative eps");
        
        std::panic::catch_unwind(|| {
            DBSCAN::new(0.0, 2);
        }).expect_err("Should panic on zero eps");
    }

    #[test]
    fn test_dbscan_invalid_min_samples() {
        std::panic::catch_unwind(|| {
            DBSCAN::new(1.0, 0);
        }).expect_err("Should panic on zero min_samples");
    }

    #[test]
    fn test_dbscan_predict_not_supported() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let mut dbscan = DBSCAN::new(1.0, 2);
        dbscan.fit(&x).unwrap();
        
        // predict() should return an error
        assert!(dbscan.predict(&x).is_err());
    }

    #[test]
    fn test_dbscan_core_sample_check() {
        let x = array![
            [1.0, 1.0],
            [1.1, 1.0],
            [1.2, 1.0],
            [10.0, 10.0] // Isolated point
        ];

        let mut dbscan = DBSCAN::new(0.5, 2);
        dbscan.fit(&x).unwrap();
        
        // First few points should be core samples
        assert_eq!(dbscan.is_core_sample(0), Some(true));
        assert_eq!(dbscan.is_core_sample(1), Some(true));
        
        // Last point should not be a core sample
        assert_eq!(dbscan.is_core_sample(3), Some(false));
    }
}