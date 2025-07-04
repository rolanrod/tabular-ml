use crate::{Matrix, Vector};
use ndarray::Dimension;

#[derive(Clone, Debug)]
pub struct Ridge {
    pub coefficients: Option<Vector>,
    pub intercept: Option<f64>,
    alpha: f64,
    fit_intercept: bool,
    normalize: bool,
}

impl Ridge {
    pub fn new() -> Self {
        Self {
            coefficients: None,
            intercept: None,
            alpha: 1.0,
            fit_intercept: true,
            normalize: false,
        }
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        if alpha < 0.0 {
            panic!("alpha must be non-negative, got {}", alpha);
        }
        self.alpha = alpha;
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
        
        let coeffs_scaled = self.solve_ridge_equation(&x_processed, &y_centered)?;
        
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
        
        let coeffs_scaled = self.solve_ridge_equation(&x_processed, y)?;
        
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
        
        for mut col in x_normalized.axis_iter_mut(ndarray::Axis(1)) {
            let std = x_std[col.raw_dim().as_array_view().iter().position(|_| true).unwrap()];
            if std > 1e-10 {
                col /= std;
            }
        }
        
        (x_normalized, x_std)
    }

    fn solve_ridge_equation(&self, x: &Matrix, y: &Vector) -> Result<Vector, String> {
        let xt = x.t();
        let xtx = xt.dot(x);
        
        let mut regularized_xtx = xtx;
        for i in 0..regularized_xtx.nrows() {
            regularized_xtx[(i, i)] += self.alpha;
        }
        
        let xty = xt.dot(y);
        
        self.solve_linear_system(&regularized_xtx, &xty)
    }

    fn solve_linear_system(&self, a: &Matrix, b: &Vector) -> Result<Vector, String> {
        use crate::Matrix;
        
        let n = a.nrows();
        let mut aug = Matrix::zeros((n, n + 1));
        
        for i in 0..n {
            for j in 0..n {
                aug[(i, j)] = a[(i, j)];
            }
            aug[(i, n)] = b[i];
        }
        
        for i in 0..n {
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug[(k, i)].abs() > aug[(max_row, i)].abs() {
                    max_row = k;
                }
            }
            
            if aug[(max_row, i)].abs() < 1e-10 {
                return Err("Matrix is singular or nearly singular".to_string());
            }
            
            if max_row != i {
                for j in 0..=n {
                    let temp = aug[(i, j)];
                    aug[(i, j)] = aug[(max_row, j)];
                    aug[(max_row, j)] = temp;
                }
            }
            
            for k in (i + 1)..n {
                let factor = aug[(k, i)] / aug[(i, i)];
                for j in i..=n {
                    aug[(k, j)] -= factor * aug[(i, j)];
                }
            }
        }
        
        let mut x = Vector::zeros(n);
        for i in (0..n).rev() {
            x[i] = aug[(i, n)];
            for j in (i + 1)..n {
                x[i] -= aug[(i, j)] * x[j];
            }
            x[i] /= aug[(i, i)];
        }
        
        Ok(x)
    }
}

impl Default for Ridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_ridge_simple() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];
        
        let mut model = Ridge::new().alpha(0.0);
        model.fit(&x, &y).unwrap();
        
        let predictions = model.predict(&x).unwrap();
        
        for (pred, actual) in predictions.iter().zip(y.iter()) {
            assert!((pred - actual).abs() < 0.1);
        }
    }

    #[test]
    fn test_ridge_with_regularization() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.1, 3.9, 6.1, 7.9];
        
        let mut model = Ridge::new().alpha(1.0);
        model.fit(&x, &y).unwrap();
        
        let score = model.score(&x, &y).unwrap();
        assert!(score > 0.8);
    }

    #[test]
    fn test_ridge_without_intercept() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];
        
        let mut model = Ridge::new().alpha(0.1).fit_intercept(false);
        model.fit(&x, &y).unwrap();
        
        assert_eq!(model.intercept.unwrap(), 0.0);
        let coeffs = model.coefficients.as_ref().unwrap();
        assert!((coeffs[0] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_ridge_multivariate() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![5.0, 8.0, 11.0, 14.0];
        
        let mut model = Ridge::new().alpha(0.5);
        model.fit(&x, &y).unwrap();
        
        let score = model.score(&x, &y).unwrap();
        assert!(score > 0.8);
    }

    #[test]
    fn test_ridge_with_normalization() {
        let x = array![[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]];
        let y = array![5.0, 8.0, 11.0, 14.0];
        
        let mut model = Ridge::new().alpha(1.0).normalize(true);
        model.fit(&x, &y).unwrap();
        
        let score = model.score(&x, &y).unwrap();
        assert!(score > 0.5);
    }

    #[test]
    fn test_ridge_invalid_alpha() {
        std::panic::catch_unwind(|| {
            Ridge::new().alpha(-1.0);
        }).expect_err("Should panic on negative alpha");
    }

    #[test]
    fn test_ridge_predict_without_fit() {
        let x = array![[1.0], [2.0]];
        let model = Ridge::new();
        
        assert!(model.predict(&x).is_err());
    }

    #[test]
    fn test_ridge_dimension_mismatch() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0, 3.0];
        
        let mut model = Ridge::new();
        assert!(model.fit(&x, &y).is_err());
    }
}