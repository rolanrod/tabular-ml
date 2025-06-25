use crate::{Matrix, Vector};

#[derive(Clone, Debug)]
pub struct ElasticNet {
    pub coefficients: Option<Vector>,
    pub intercept: Option<f64>,
    alpha: f64,
    l1_ratio: f64,
    fit_intercept: bool,
    normalize: bool,
    max_iter: usize,
    tolerance: f64,
}

impl ElasticNet {
    pub fn new() -> Self {
        Self {
            coefficients: None,
            intercept: None,
            alpha: 1.0,
            l1_ratio: 0.5,
            fit_intercept: true,
            normalize: false,
            max_iter: 1000,
            tolerance: 1e-4,
        }
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        if alpha < 0.0 {
            panic!("alpha must be non-negative, got {}", alpha);
        }
        self.alpha = alpha;
        self
    }

    pub fn l1_ratio(mut self, l1_ratio: f64) -> Self {
        if l1_ratio < 0.0 || l1_ratio > 1.0 {
            panic!("l1_ratio must be between 0 and 1, got {}", l1_ratio);
        }
        self.l1_ratio = l1_ratio;
        self
    }

    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn fit(&mut self, x: &Matrix, y: &Vector) -> Result<(), String> {
        if x.nrows() != y.len() {
            return Err("Number of samples in X and y must match".to_string());
        }

        if x.nrows() == 0 {
            return Err("X must have at least one sample".to_string());
        }

        let (coeffs, intercept) = if self.fit_intercept {
            self.fit_with_intercept(x, y)?
        } else {
            (self.fit_without_intercept(x, y)?, 0.0)
        };

        self.coefficients = Some(coeffs);
        self.intercept = Some(intercept);
        Ok(())
    }

    pub fn predict(&self, x: &Matrix) -> Result<Vector, String> {
        let coeffs = self.coefficients.as_ref()
            .ok_or("Model not fitted. Call fit() first.")?;
        let intercept = self.intercept.unwrap_or(0.0);

        if x.ncols() != coeffs.len() {
            return Err(format!(
                "Number of features in X ({}) doesn't match training data ({})",
                x.ncols(),
                coeffs.len()
            ));
        }

        let predictions = x.dot(coeffs) + intercept;
        Ok(predictions)
    }

    pub fn score(&self, x: &Matrix, y: &Vector) -> Result<f64, String> {
        let y_pred = self.predict(x)?;
        crate::metrics::r2_score(y, &y_pred)
    }

    fn fit_with_intercept(&self, x: &Matrix, y: &Vector) -> Result<(Vector, f64), String> {
        let y_mean = y.mean().unwrap();
        let x_means = x.mean_axis(ndarray::Axis(0)).unwrap();
        
        let mut x_centered = x.clone();
        for mut row in x_centered.axis_iter_mut(ndarray::Axis(0)) {
            row -= &x_means;
        }
        
        let y_centered = y - y_mean;
        
        let (x_processed, x_scales) = if self.normalize {
            self.normalize_features(&x_centered)
        } else {
            (x_centered, Vector::ones(x.ncols()))
        };
        
        let coeffs_scaled = self.coordinate_descent(&x_processed, &y_centered)?;
        
        let coeffs = if self.normalize {
            &coeffs_scaled / &x_scales
        } else {
            coeffs_scaled
        };
        
        let intercept = y_mean - coeffs.dot(&x_means);
        
        Ok((coeffs, intercept))
    }

    fn fit_without_intercept(&self, x: &Matrix, y: &Vector) -> Result<Vector, String> {
        let (x_processed, x_scales) = if self.normalize {
            self.normalize_features(x)
        } else {
            (x.clone(), Vector::ones(x.ncols()))
        };
        
        let coeffs_scaled = self.coordinate_descent(&x_processed, y)?;
        
        let coeffs = if self.normalize {
            &coeffs_scaled / &x_scales
        } else {
            coeffs_scaled
        };
        
        Ok(coeffs)
    }

    fn normalize_features(&self, x: &Matrix) -> (Matrix, Vector) {
        let x_std = x.std_axis(ndarray::Axis(0), 0.0);
        let mut x_normalized = x.clone();
        
        for j in 0..x.ncols() {
            let std = x_std[j];
            if std > 1e-10 {
                for i in 0..x.nrows() {
                    x_normalized[(i, j)] /= std;
                }
            }
        }
        
        (x_normalized, x_std)
    }

    fn coordinate_descent(&self, x: &Matrix, y: &Vector) -> Result<Vector, String> {
        let n_features = x.ncols();
        let n_samples = x.nrows() as f64;
        let mut beta = Vector::zeros(n_features);
        
        let x_norms: Vector = (0..n_features)
            .map(|j| x.column(j).mapv(|x| x * x).sum())
            .collect();
        
        let l1_penalty = self.alpha * self.l1_ratio;
        let l2_penalty = self.alpha * (1.0 - self.l1_ratio);
        
        for _ in 0..self.max_iter {
            let beta_old = beta.clone();
            
            for j in 0..n_features {
                if x_norms[j] < 1e-10 {
                    continue;
                }
                
                let r_j = self.compute_residual_j(x, y, &beta, j);
                let z_j = x.column(j).dot(&r_j) / n_samples + beta[j];
                
                let denominator = x_norms[j] / n_samples + l2_penalty;
                let numerator_threshold = l1_penalty / denominator;
                
                beta[j] = self.soft_threshold(z_j / denominator, numerator_threshold);
            }
            
            let diff = (&beta - &beta_old).mapv(|x| x.abs()).sum();
            if diff < self.tolerance {
                break;
            }
        }
        
        Ok(beta)
    }

    fn compute_residual_j(&self, x: &Matrix, y: &Vector, beta: &Vector, j: usize) -> Vector {
        let mut r = y.clone();
        
        for k in 0..x.ncols() {
            if k != j {
                for i in 0..x.nrows() {
                    r[i] -= x[(i, k)] * beta[k];
                }
            }
        }
        
        r
    }

    fn soft_threshold(&self, z: f64, gamma: f64) -> f64 {
        if z > gamma {
            z - gamma
        } else if z < -gamma {
            z + gamma
        } else {
            0.0
        }
    }

    pub fn l1_penalty(&self) -> f64 {
        self.alpha * self.l1_ratio
    }

    pub fn l2_penalty(&self) -> f64 {
        self.alpha * (1.0 - self.l1_ratio)
    }
}

impl Default for ElasticNet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_elastic_net_simple() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];
        
        let mut model = ElasticNet::new().alpha(0.01).l1_ratio(0.5);
        model.fit(&x, &y).unwrap();
        
        let predictions = model.predict(&x).unwrap();
        let score = model.score(&x, &y).unwrap();
        
        assert!(score > 0.8);
    }

    #[test]
    fn test_elastic_net_l1_ratio_extremes() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];
        
        let mut lasso_like = ElasticNet::new().alpha(0.01).l1_ratio(1.0);
        lasso_like.fit(&x, &y).unwrap();
        
        let mut ridge_like = ElasticNet::new().alpha(0.01).l1_ratio(0.0);
        ridge_like.fit(&x, &y).unwrap();
        
        let lasso_score = lasso_like.score(&x, &y).unwrap();
        let ridge_score = ridge_like.score(&x, &y).unwrap();
        
        assert!(lasso_score > 0.5);
        assert!(ridge_score > 0.5);
    }

    #[test]
    fn test_elastic_net_sparsity() {
        let x = array![
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0];
        
        let mut model = ElasticNet::new().alpha(0.1).l1_ratio(0.8);
        model.fit(&x, &y).unwrap();
        
        let coeffs = model.coefficients.as_ref().unwrap();
        
        assert!(coeffs[0].abs() > 0.1);
        assert!(coeffs[1].abs() < 0.1);
        assert!(coeffs[2].abs() < 0.1);
    }

    #[test]
    fn test_elastic_net_without_intercept() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];
        
        let mut model = ElasticNet::new()
            .alpha(0.01)
            .l1_ratio(0.5)
            .fit_intercept(false);
        model.fit(&x, &y).unwrap();
        
        assert_eq!(model.intercept.unwrap(), 0.0);
        let coeffs = model.coefficients.as_ref().unwrap();
        assert!((coeffs[0] - 2.0).abs() < 0.2);
    }

    #[test]
    fn test_elastic_net_multivariate() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0]
        ];
        let y = array![5.0, 8.0, 11.0, 14.0, 17.0];
        
        let mut model = ElasticNet::new().alpha(0.1).l1_ratio(0.5);
        model.fit(&x, &y).unwrap();
        
        let score = model.score(&x, &y).unwrap();
        assert!(score > 0.7);
    }

    #[test]
    fn test_elastic_net_penalty_methods() {
        let model = ElasticNet::new().alpha(1.0).l1_ratio(0.7);
        
        assert!((model.l1_penalty() - 0.7).abs() < 1e-10);
        assert!((model.l2_penalty() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_elastic_net_high_regularization() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];
        
        let mut model = ElasticNet::new().alpha(100.0).l1_ratio(0.5);
        model.fit(&x, &y).unwrap();
        
        let coeffs = model.coefficients.as_ref().unwrap();
        assert!(coeffs[0].abs() < 0.1);
    }

    #[test]
    fn test_elastic_net_invalid_alpha() {
        std::panic::catch_unwind(|| {
            ElasticNet::new().alpha(-1.0);
        }).expect_err("Should panic on negative alpha");
    }

    #[test]
    fn test_elastic_net_invalid_l1_ratio() {
        std::panic::catch_unwind(|| {
            ElasticNet::new().l1_ratio(-0.1);
        }).expect_err("Should panic on negative l1_ratio");
        
        std::panic::catch_unwind(|| {
            ElasticNet::new().l1_ratio(1.1);
        }).expect_err("Should panic on l1_ratio > 1");
    }

    #[test]
    fn test_elastic_net_predict_without_fit() {
        let x = array![[1.0], [2.0]];
        let model = ElasticNet::new();
        
        assert!(model.predict(&x).is_err());
    }

    #[test]
    fn test_elastic_net_dimension_mismatch() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0, 3.0];
        
        let mut model = ElasticNet::new();
        assert!(model.fit(&x, &y).is_err());
    }
}