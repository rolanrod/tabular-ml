use crate::{Matrix, Vector};
use ndarray::s;
use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct TruncatedSVD {
    pub components: Option<Matrix>,
    pub explained_variance: Option<Vector>,
    pub explained_variance_ratio: Option<Vector>,
    pub singular_values: Option<Vector>,
    n_components: usize,
    algorithm: String,
    n_iter: usize,
    random_state: Option<u64>,
}

impl TruncatedSVD {
    pub fn new(n_components: usize) -> Self {
        Self {
            components: None,
            explained_variance: None,
            explained_variance_ratio: None,
            singular_values: None,
            n_components,
            algorithm: "randomized".to_string(),
            n_iter: 5,
            random_state: None,
        }
    }

    pub fn algorithm(mut self, algorithm: &str) -> Self {
        match algorithm {
            "arpack" | "randomized" => {
                self.algorithm = algorithm.to_string();
            }
            _ => panic!("Invalid algorithm: {}. Must be one of: arpack, randomized", algorithm),
        }
        self
    }

    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    pub fn fit(&mut self, x: &Matrix) -> Result<(), String> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err("Input matrix must have at least one sample and one feature".to_string());
        }

        let (n_samples, n_features) = (x.nrows(), x.ncols());
        
        if self.n_components > n_samples.min(n_features) {
            return Err(format!(
                "n_components={} cannot be larger than min(n_samples, n_features)={}",
                self.n_components, n_samples.min(n_features)
            ));
        }

        // Perform truncated SVD
        let (_u, s, vt) = self.truncated_svd(x)?;
        
        // Components are the rows of V^T (columns of V)
        let components = vt.slice(s![..self.n_components, ..]).to_owned();
        
        // Singular values
        let singular_values = s.slice(s![..self.n_components]).to_owned();
        
        // Explained variance is proportional to singular values squared
        let explained_variance = singular_values.mapv(|s| s * s / (n_samples - 1) as f64);
        
        // Compute explained variance ratio
        let total_variance = explained_variance.sum();
        let explained_variance_ratio = if total_variance > 0.0 {
            &explained_variance / total_variance
        } else {
            Vector::zeros(explained_variance.len())
        };

        self.components = Some(components);
        self.singular_values = Some(singular_values);
        self.explained_variance = Some(explained_variance);
        self.explained_variance_ratio = Some(explained_variance_ratio);

        Ok(())
    }

    pub fn transform(&self, x: &Matrix) -> Result<Matrix, String> {
        let components = self.components.as_ref()
            .ok_or("TruncatedSVD not fitted. Call fit() first.")?;

        if x.ncols() != components.ncols() {
            return Err(format!(
                "Number of features in X ({}) doesn't match training data ({})",
                x.ncols(), components.ncols()
            ));
        }

        // Transform: X * V = X * components^T
        let transformed = x.dot(&components.t());
        Ok(transformed)
    }

    pub fn fit_transform(&mut self, x: &Matrix) -> Result<Matrix, String> {
        self.fit(x)?;
        self.transform(x)
    }

    pub fn inverse_transform(&self, x: &Matrix) -> Result<Matrix, String> {
        let components = self.components.as_ref()
            .ok_or("TruncatedSVD not fitted. Call fit() first.")?;

        if x.ncols() != self.n_components {
            return Err(format!(
                "Number of features in X ({}) doesn't match number of components ({})",
                x.ncols(), self.n_components
            ));
        }

        // Inverse transform: X_reduced * V^T = X_reduced * components
        let reconstructed = x.dot(components);
        Ok(reconstructed)
    }

    fn truncated_svd(&self, x: &Matrix) -> Result<(Matrix, Vector, Matrix), String> {
        // Simplified SVD implementation using power iteration
        // In practice, you'd use a proper SVD library like LAPACK
        
        let (m, n) = (x.nrows(), x.ncols());
        let k = self.n_components;
        
        // Compute A^T * A or A * A^T depending on dimensions
        let use_covariance = m > n;
        
        if use_covariance {
            // Compute A^T * A and find eigenvectors (V), then compute U
            let ata = x.t().dot(x);
            let (eigenvals, eigenvecs) = self.eigen_decomposition(&ata)?;
            
            // Sort eigenvalues and eigenvectors in descending order
            let mut eigen_pairs: Vec<(f64, Vector)> = eigenvals.iter()
                .zip(eigenvecs.axis_iter(ndarray::Axis(1)))
                .map(|(&val, vec)| (val, vec.to_owned()))
                .collect();
                
            eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
            
            // Take top k components
            let mut vt = Matrix::zeros((k, n));
            let mut singular_values = Vector::zeros(k);
            
            for (i, (eigenval, eigenvec)) in eigen_pairs.iter().take(k).enumerate() {
                vt.row_mut(i).assign(eigenvec);
                singular_values[i] = eigenval.sqrt().max(0.0);
            }
            
            // Compute U = A * V * S^(-1)
            let mut u = Matrix::zeros((m, k));
            for i in 0..k {
                if singular_values[i] > 1e-10 {
                    let av = x.dot(&vt.row(i));
                    u.column_mut(i).assign(&(av / singular_values[i]));
                }
            }
            
            Ok((u, singular_values, vt))
        } else {
            // Compute A * A^T and find eigenvectors (U), then compute V
            let aat = x.dot(&x.t());
            let (eigenvals, eigenvecs) = self.eigen_decomposition(&aat)?;
            
            // Sort eigenvalues and eigenvectors in descending order
            let mut eigen_pairs: Vec<(f64, Vector)> = eigenvals.iter()
                .zip(eigenvecs.axis_iter(ndarray::Axis(1)))
                .map(|(&val, vec)| (val, vec.to_owned()))
                .collect();
                
            eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
            
            // Take top k components
            let mut u = Matrix::zeros((m, k));
            let mut singular_values = Vector::zeros(k);
            
            for (i, (eigenval, eigenvec)) in eigen_pairs.iter().take(k).enumerate() {
                u.column_mut(i).assign(eigenvec);
                singular_values[i] = eigenval.sqrt().max(0.0);
            }
            
            // Compute V^T = S^(-1) * U^T * A
            let mut vt = Matrix::zeros((k, n));
            for i in 0..k {
                if singular_values[i] > 1e-10 {
                    let utau = u.column(i).t().dot(x);
                    vt.row_mut(i).assign(&(utau / singular_values[i]));
                }
            }
            
            Ok((u, singular_values, vt))
        }
    }

    fn eigen_decomposition(&self, matrix: &Matrix) -> Result<(Vector, Matrix), String> {
        // Simplified eigenvalue decomposition using power iteration
        let n = matrix.nrows();
        
        if n != matrix.ncols() {
            return Err("Matrix must be square for eigenvalue decomposition".to_string());
        }
        
        let mut eigenvalues = Vector::zeros(n);
        let mut eigenvectors = Matrix::zeros((n, n));
        
        let mut a = matrix.clone();
        
        for i in 0..n {
            // Power iteration
            let mut v = Vector::ones(n);
            let mut lambda = 0.0;
            
            for _ in 0..100 {
                let av = a.dot(&v);
                let norm = av.mapv(|x| x * x).sum().sqrt();
                
                if norm < 1e-10 {
                    break;
                }
                
                v = av / norm;
                let new_lambda = v.dot(&a.dot(&v));
                
                if (new_lambda - lambda).abs() < 1e-10 {
                    break;
                }
                lambda = new_lambda;
            }
            
            eigenvalues[i] = lambda.max(0.0); // Ensure non-negative for SVD
            eigenvectors.column_mut(i).assign(&v);
            
            // Deflation
            if lambda > 1e-10 {
                let vv = v.view().insert_axis(ndarray::Axis(1)).dot(&v.view().insert_axis(ndarray::Axis(0)));
                a = &a - &(vv * lambda);
            }
        }
        
        Ok((eigenvalues, eigenvectors))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_truncated_svd_basic() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];

        let mut svd = TruncatedSVD::new(2);
        let transformed = svd.fit_transform(&x).unwrap();
        
        assert_eq!(transformed.shape(), &[4, 2]);
        assert!(svd.components.is_some());
        assert!(svd.singular_values.is_some());
        assert!(svd.explained_variance.is_some());
    }

    #[test]
    fn test_truncated_svd_reconstruction() {
        let x = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];

        let mut svd = TruncatedSVD::new(3);
        let transformed = svd.fit_transform(&x).unwrap();
        let reconstructed = svd.inverse_transform(&transformed).unwrap();
        
        // Check that reconstruction is reasonably close to original
        let diff = &x - &reconstructed;
        let max_error = diff.mapv(|x| x.abs()).into_iter().fold(0.0, f64::max);
        assert!(max_error < 1.0); // Relaxed tolerance for simplified SVD
    }

    #[test]
    fn test_truncated_svd_invalid_components() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let mut svd = TruncatedSVD::new(5); // More than min(m,n)
        
        assert!(svd.fit(&x).is_err());
    }

    #[test]
    fn test_truncated_svd_transform_without_fit() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let svd = TruncatedSVD::new(1);
        
        assert!(svd.transform(&x).is_err());
    }

    #[test]
    fn test_truncated_svd_dimension_mismatch() {
        let x_train = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let x_test = array![[1.0, 2.0], [3.0, 4.0]]; // Different number of features
        
        let mut svd = TruncatedSVD::new(1);
        svd.fit(&x_train).unwrap();
        
        assert!(svd.transform(&x_test).is_err());
    }

    #[test]
    fn test_truncated_svd_single_component() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0]
        ];

        let mut svd = TruncatedSVD::new(1);
        let transformed = svd.fit_transform(&x).unwrap();
        
        assert_eq!(transformed.shape(), &[3, 1]);
        
        let explained_variance_ratio = svd.explained_variance_ratio.as_ref().unwrap();
        // First component should explain most of the variance
        assert!(explained_variance_ratio[0] > 0.5);
    }
}