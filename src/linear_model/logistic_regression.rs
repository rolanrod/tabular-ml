use crate::{Matrix, Vector};

#[derive(Clone, Debug)]
pub struct LogisticRegression {
    pub coefficients: Option<Vector>,
    pub intercept: Option<f64>,
    fit_intercept: bool,
    learning_rate: f64,
    max_iterations: usize,
    tolerance: f64,
}

impl LogisticRegression {
    pub fn new() -> Self {
        Self {
            coefficients: None,
            intercept: None,
            fit_intercept: true,
            learning_rate: 0.01,
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }

    pub fn with_params(learning_rate: f64, max_iterations: usize, fit_intercept: bool) -> Self {
        Self {
            coefficients: None,
            intercept: None,
            fit_intercept,
            learning_rate,
            max_iterations,
            tolerance: 1e-6,
        }
    }

    pub fn fit(&mut self, x: &Matrix, y: &Vector) -> Result<(), String> {
        if x.nrows() != y.len() {
            return Err("Number of samples in X and y must match".to_string());
        }

        self.validate_labels(y)?;

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
        let probabilities = self.predict_proba(x)?;
        let predictions = probabilities.mapv(|p| if p >= 0.5 { 1.0 } else { 0.0 });
        Ok(predictions)
    }

    pub fn predict_proba(&self, x: &Matrix) -> Result<Vector, String> {
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

        let linear_combination = x.dot(coeffs) + intercept;
        let probabilities = linear_combination.mapv(|z| Self::sigmoid(z));
        Ok(probabilities)
    }

    pub fn score(&self, x: &Matrix, y: &Vector) -> Result<f64, String> {
        let predictions = self.predict(x)?;
        let accuracy = predictions.iter()
            .zip(y.iter())
            .map(|(pred, actual)| if (pred - actual).abs() < 1e-10 { 1.0 } else { 0.0 })
            .sum::<f64>() / y.len() as f64;
        Ok(accuracy)
    }

    fn sigmoid(z: f64) -> f64 {
        if z > 500.0 {
            1.0
        } else if z < -500.0 {
            0.0
        } else {
            1.0 / (1.0 + (-z).exp())
        }
    }

    fn validate_labels(&self, y: &Vector) -> Result<(), String> {
        for &label in y.iter() {
            if label != 0.0 && label != 1.0 {
                return Err("Labels must be 0 or 1 for binary classification".to_string());
            }
        }
        Ok(())
    }

    fn fit_with_intercept(&self, x: &Matrix, y: &Vector) -> Result<(Vector, f64), String> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        
        let mut x_with_intercept = Matrix::ones((n_samples, n_features + 1));
        x_with_intercept.slice_mut(ndarray::s![.., 1..]).assign(x);

        let coeffs_with_intercept = self.gradient_descent(&x_with_intercept, y)?;
        
        let intercept = coeffs_with_intercept[0];
        let coeffs = coeffs_with_intercept.slice(ndarray::s![1..]).to_owned();
        
        Ok((coeffs, intercept))
    }

    fn fit_without_intercept(&self, x: &Matrix, y: &Vector) -> Result<Vector, String> {
        self.gradient_descent(x, y)
    }

    fn gradient_descent(&self, x: &Matrix, y: &Vector) -> Result<Vector, String> {
        let n_features = x.ncols();
        let n_samples = x.nrows() as f64;
        let mut weights = Vector::zeros(n_features);
        
        let mut prev_cost = f64::INFINITY;
        let mut no_improvement_count = 0;
        
        for _iteration in 0..self.max_iterations {
            let linear_combination = x.dot(&weights);
            let predictions = linear_combination.mapv(|z| Self::sigmoid(z));
            
            let cost = self.logistic_loss(y, &predictions);
            
            if cost.is_nan() || cost.is_infinite() {
                return Err("Gradient descent diverged".to_string());
            }
            
            if (prev_cost - cost).abs() < self.tolerance {
                no_improvement_count += 1;
                if no_improvement_count > 10 {
                    break;
                }
            } else {
                no_improvement_count = 0;
            }
            
            let error = &predictions - y;
            let gradient = x.t().dot(&error) / n_samples;
            weights = &weights - &gradient * self.learning_rate;
            
            prev_cost = cost;
        }
        
        Ok(weights)
    }

    fn logistic_loss(&self, y_true: &Vector, y_pred: &Vector) -> f64 {
        let epsilon = 1e-15;
        let clipped_pred = y_pred.mapv(|p| p.max(epsilon).min(1.0 - epsilon));
        
        let loss = y_true.iter()
            .zip(clipped_pred.iter())
            .map(|(&y, &p)| -y * p.ln() - (1.0 - y) * (1.0 - p).ln())
            .sum::<f64>();
        
        loss / y_true.len() as f64
    }
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_logistic_regression_simple() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];
        
        let mut model = LogisticRegression::new();
        model.fit(&x, &y).unwrap();
        
        let predictions = model.predict(&x).unwrap();
        let probabilities = model.predict_proba(&x).unwrap();
        
        assert_eq!(predictions.len(), 4);
        assert_eq!(probabilities.len(), 4);
        
        assert!(probabilities[0] < 0.5);
        assert!(probabilities[3] > 0.5);
    }

    #[test]
    fn test_logistic_regression_score() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];
        
        let mut model = LogisticRegression::new();
        model.fit(&x, &y).unwrap();
        
        let score = model.score(&x, &y).unwrap();
        assert!(score > 0.5);
    }

    #[test]
    fn test_logistic_regression_invalid_labels() {
        let x = array![[1.0], [2.0]];
        let y = array![0.5, 2.0];
        
        let mut model = LogisticRegression::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_logistic_regression_predict_without_fit() {
        let x = array![[1.0], [2.0]];
        let model = LogisticRegression::new();
        
        assert!(model.predict(&x).is_err());
        assert!(model.predict_proba(&x).is_err());
    }

    #[test]
    fn test_sigmoid_function() {
        assert!((LogisticRegression::sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(LogisticRegression::sigmoid(1000.0) > 0.99);
        assert!(LogisticRegression::sigmoid(-1000.0) < 0.01);
    }
}