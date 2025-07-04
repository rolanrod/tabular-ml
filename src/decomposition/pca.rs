use crate::{Matrix, Vector};
use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct PCA {
    pub components: Option<Matrix>,
    pub explained_variance: Option<Vector>,
    pub explained_variance_ratio: Option<Vector>,
    pub mean: Option<Vector>,
    n_components: Option<usize>,
    svd_solver: String,
}

impl PCA {
    pub fn new() -> Self {
        Self {
            components: None,
            explained_variance: None,
            explained_variance_ratio: None,
            mean: None,
            n_components: None,
            svd_solver: "auto".to_string(),
        }
    }

    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = Some(n_components);
        self
    }

    pub fn svd_solver(mut self, solver: &str) -> Self {
        match solver {
            "auto" | "full" | "arpack" | "randomized" => {
                self.svd_solver = solver.to_string();
            }
            _ => panic!("Invalid svd_solver: {}. Must be one of: auto, full, arpack, randomized", solver),
        }
        self
    }

    pub fn fit(&mut self, x: &Matrix) -> Result<(), String> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err("Input matrix must have at least one sample and one feature".to_string());
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Determine number of components
        let n_components = self.n_components.unwrap_or(n_features.min(n_samples));
        
        if n_components > n_features.min(n_samples) {
            return Err(format!(
                "n_components={} cannot be larger than min(n_samples, n_features)={}",
                n_components, n_features.min(n_samples)
            ));
        }

        // Center the data
        let mean = x.mean_axis(ndarray::Axis(0)).unwrap();
        let x_centered = x - &mean.view().insert_axis(ndarray::Axis(0));

        // Compute covariance matrix or use SVD directly
        let (components, explained_variance) = if n_samples > n_features {
            // More samples than features: use covariance matrix approach
            self.fit_covariance(&x_centered, n_components)?
        } else {
            // More features than samples: use SVD approach
            self.fit_svd(&x_centered, n_components)?
        };

        // Compute explained variance ratio
        let total_variance = explained_variance.sum();
        let explained_variance_ratio = if total_variance > 0.0 {
            &explained_variance / total_variance
        } else {
            Vector::zeros(explained_variance.len())
        };

        self.components = Some(components);
        self.explained_variance = Some(explained_variance);
        self.explained_variance_ratio = Some(explained_variance_ratio);
        self.mean = Some(mean);

        Ok(())
    }

    pub fn transform(&self, x: &Matrix) -> Result<Matrix, String> {
        let components = self.components.as_ref()
            .ok_or("PCA not fitted. Call fit() first.")?;
        let mean = self.mean.as_ref()
            .ok_or("PCA not fitted. Call fit() first.")?;

        if x.ncols() != mean.len() {
            return Err(format!(
                "Number of features in X ({}) doesn't match training data ({})",
                x.ncols(), mean.len()
            ));
        }

        // Center the data
        let x_centered = x - &mean.view().insert_axis(ndarray::Axis(0));
        
        // Project onto principal components
        let transformed = x_centered.dot(&components.t());
        
        Ok(transformed)
    }

    pub fn fit_transform(&mut self, x: &Matrix) -> Result<Matrix, String> {
        self.fit(x)?;
        self.transform(x)
    }

    pub fn inverse_transform(&self, x: &Matrix) -> Result<Matrix, String> {
        let components = self.components.as_ref()
            .ok_or("PCA not fitted. Call fit() first.")?;
        let mean = self.mean.as_ref()
            .ok_or("PCA not fitted. Call fit() first.")?;

        if x.ncols() != components.nrows() {
            return Err(format!(
                "Number of features in X ({}) doesn't match number of components ({})",
                x.ncols(), components.nrows()
            ));
        }

        // Project back to original space
        let x_reconstructed = x.dot(components) + &mean.view().insert_axis(ndarray::Axis(0));
        
        Ok(x_reconstructed)
    }

    fn fit_covariance(&self, x_centered: &Matrix, n_components: usize) -> Result<(Matrix, Vector), String> {
        let n_samples = x_centered.nrows() as f64;
        
        // Compute covariance matrix
        let cov = x_centered.t().dot(x_centered) / (n_samples - 1.0);
        
        // Compute eigenvalues and eigenvectors
        let (eigenvalues, eigenvectors) = self.eigen_decomposition(&cov)?;
        
        // Sort by eigenvalues (descending)
        let mut eigen_pairs: Vec<(f64, Vector)> = eigenvalues.iter()
            .zip(eigenvectors.axis_iter(ndarray::Axis(1)))
            .map(|(&val, vec)| (val, vec.to_owned()))
            .collect();
            
        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        
        // Select top n_components
        let selected_eigenvalues: Vector = eigen_pairs.iter()
            .take(n_components)
            .map(|(val, _)| *val)
            .collect::<Vec<f64>>()
            .into();
            
        let mut components = Matrix::zeros((n_components, x_centered.ncols()));
        for (i, (_, eigenvec)) in eigen_pairs.iter().take(n_components).enumerate() {
            components.row_mut(i).assign(eigenvec);
        }
        
        Ok((components, selected_eigenvalues))
    }

    fn fit_svd(&self, x_centered: &Matrix, n_components: usize) -> Result<(Matrix, Vector), String> {
        // For SVD approach: X = U * S * V^T
        // Principal components are the columns of V
        // Explained variance is S^2 / (n_samples - 1)
        
        let n_samples = x_centered.nrows() as f64;
        
        // Use simple power iteration for SVD (simplified implementation)
        // In a real implementation, you'd use a proper SVD library
        let cov = x_centered.t().dot(x_centered) / (n_samples - 1.0);
        let x_t = x_centered.t().to_owned();
        let x_scaled = x_t / (n_samples - 1.0).sqrt();
        self.fit_covariance(&x_scaled, n_components)
    }

    fn eigen_decomposition(&self, matrix: &Matrix) -> Result<(Vector, Matrix), String> {
        // Simplified eigenvalue decomposition using power iteration
        // This is a basic implementation - in practice you'd use LAPACK
        let n = matrix.nrows();
        
        if n != matrix.ncols() {
            return Err("Matrix must be square for eigenvalue decomposition".to_string());
        }
        
        let mut eigenvalues = Vector::zeros(n);
        let mut eigenvectors = Matrix::zeros((n, n));
        
        // Use power iteration to find dominant eigenvalues/eigenvectors
        let mut a = matrix.clone();
        
        for i in 0..n {
            // Power iteration
            let mut v = Vector::ones(n);
            let mut lambda = 0.0;
            
            for _ in 0..100 { // max iterations
                let av = a.dot(&v);
                let new_lambda = v.dot(&av);
                let norm = av.mapv(|x| x * x).sum().sqrt();
                
                if norm < 1e-10 {
                    break;
                }
                
                v = av / norm;
                
                if (new_lambda - lambda).abs() < 1e-10 {
                    break;
                }
                lambda = new_lambda / v.dot(&v);
            }
            
            eigenvalues[i] = lambda;
            eigenvectors.column_mut(i).assign(&v);
            
            // Deflation: remove the found eigenvalue/eigenvector
            let vv = v.view().insert_axis(ndarray::Axis(1)).dot(&v.view().insert_axis(ndarray::Axis(0)));
            a = &a - &(vv * lambda);
        }
        
        Ok((eigenvalues, eigenvectors))
    }

    pub fn get_covariance(&self) -> Option<Matrix> {
        // Return the covariance matrix if available
        None // Would need to store this during fit
    }

    pub fn score(&self, x: &Matrix) -> Result<f64, String> {
        // Return the average log-likelihood of the data
        let transformed = self.transform(x)?;
        let reconstructed = self.inverse_transform(&transformed)?;
        
        let diff = x - &reconstructed;
        let mse = diff.mapv(|x| x * x).mean().unwrap();
        
        // Return negative MSE as score (higher is better)
        Ok(-mse)
    }
}

impl Default for PCA {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_pca_basic() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];

        let mut pca = PCA::new().n_components(2);
        let transformed = pca.fit_transform(&x).unwrap();
        
        assert_eq!(transformed.shape(), &[4, 2]);
        assert!(pca.components.is_some());
        assert!(pca.explained_variance.is_some());
        assert!(pca.explained_variance_ratio.is_some());
        assert!(pca.mean.is_some());
    }

    #[test]
    fn test_pca_reconstruction() {
        let x = array![
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0]
        ];

        let mut pca = PCA::new().n_components(2);
        let transformed = pca.fit_transform(&x).unwrap();
        let reconstructed = pca.inverse_transform(&transformed).unwrap();
        
        // Check that reconstruction is close to original
        let diff = &x - &reconstructed;
        let max_error = diff.mapv(|x| x.abs()).into_iter().fold(0.0, f64::max);
        assert!(max_error < 1e-10);
    }

    #[test]
    fn test_pca_explained_variance() {
        let x = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0]
        ];

        let mut pca = PCA::new();
        pca.fit(&x).unwrap();
        
        let explained_variance_ratio = pca.explained_variance_ratio.as_ref().unwrap();
        let total_ratio: f64 = explained_variance_ratio.sum();
        
        // Total explained variance ratio should be close to 1.0
        assert!((total_ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pca_invalid_components() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let mut pca = PCA::new().n_components(5); // More than features
        
        assert!(pca.fit(&x).is_err());
    }

    #[test]
    fn test_pca_transform_without_fit() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let pca = PCA::new();
        
        assert!(pca.transform(&x).is_err());
    }

    #[test]
    fn test_pca_dimension_mismatch() {
        let x_train = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let x_test = array![[1.0, 2.0], [3.0, 4.0]]; // Different number of features
        
        let mut pca = PCA::new();
        pca.fit(&x_train).unwrap();
        
        assert!(pca.transform(&x_test).is_err());
    }

    #[test]
    fn test_pca_single_component() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0]
        ];

        let mut pca = PCA::new().n_components(1);
        let transformed = pca.fit_transform(&x).unwrap();
        
        assert_eq!(transformed.shape(), &[3, 1]);
        
        let explained_variance_ratio = pca.explained_variance_ratio.as_ref().unwrap();
        // First component should explain most of the variance
        assert!(explained_variance_ratio[0] > 0.9);
    }
}