use crate::{Matrix, Vector};

#[derive(Clone, Debug)]
pub struct LinearRegression {
    pub coefficients: Option<Vector>,
    pub intercept: Option<f64>,
    fit_intercept: bool,
}

impl LinearRegression {
    pub fn new() -> Self {
        Self {
            coefficients: None,
            intercept: None,
            fit_intercept: true,
        }
    }

    pub fn with_intercept(fit_intercept: bool) -> Self {
        Self {
            coefficients: None,
            intercept: None,
            fit_intercept,
        }
    }

    pub fn fit(&mut self, x: &Matrix, y: &Vector) -> Result<(), String> {
        if x.nrows() != y.len() {
            return Err("Number of samples in X and y must match".to_string());
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
        
        let coeffs = self.solve_normal_equation(&x_centered, &y_centered)?;
        let intercept = y_mean - coeffs.dot(&x_means);
        
        Ok((coeffs, intercept))
    }

    fn fit_without_intercept(&self, x: &Matrix, y: &Vector) -> Result<Vector, String> {
        self.solve_normal_equation(x, y)
    }

    fn solve_normal_equation(&self, x: &Matrix, y: &Vector) -> Result<Vector, String> {
        match self.analytical_solution(x, y) {
            Ok(coeffs) => Ok(coeffs),
            Err(_) => self.gradient_descent(x, y),
        }
    }

    fn analytical_solution(&self, x: &Matrix, y: &Vector) -> Result<Vector, String> {
        if x.ncols() == 1 {
            let xx = (0..x.nrows()).map(|i| x[(i, 0)] * x[(i, 0)]).sum::<f64>();
            let xy = (0..x.nrows()).map(|i| x[(i, 0)] * y[i]).sum::<f64>();
            
            if xx.abs() < 1e-10 {
                return Err("Singular matrix".to_string());
            }
            
            Ok(Vector::from(vec![xy / xx]))
        } else {
            Err("Multi-dimensional analytical solution not implemented".to_string())
        }
    }

    fn gradient_descent(&self, x: &Matrix, y: &Vector) -> Result<Vector, String> {
        let n_features = x.ncols();
        let n_samples = x.nrows() as f64;
        let mut weights = Vector::zeros(n_features);
        
        let x_std = x.std_axis(ndarray::Axis(0), 0.0);
        let y_std = y.std(0.0);
        
        let scale_factor = if y_std > 0.0 { 
            x_std.iter().map(|&s| if s > 0.0 { s / y_std } else { 1.0 }).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
        } else { 
            1.0 
        };
        
        let mut learning_rate = 0.01 / scale_factor;
        let max_iterations = 50000;
        let tolerance = 1e-10;
        
        let mut prev_cost = f64::INFINITY;
        let mut no_improvement_count = 0;
        
        for iteration in 0..max_iterations {
            let predictions = x.dot(&weights);
            let error = &predictions - y;
            let cost = error.mapv(|x| x * x).sum() / (2.0 * n_samples);
            
            if cost.is_nan() || cost.is_infinite() {
                return Err("Gradient descent diverged".to_string());
            }
            
            if cost < tolerance {
                break;
            }
            
            if (prev_cost - cost).abs() < tolerance {
                no_improvement_count += 1;
                if no_improvement_count > 100 {
                    break;
                }
            } else {
                no_improvement_count = 0;
            }
            
            if cost > prev_cost && iteration > 10 {
                learning_rate *= 0.95;
            } else if iteration > 100 && (prev_cost - cost) > 0.01 {
                learning_rate *= 1.001;
            }
            
            let gradient = x.t().dot(&error) / n_samples;
            weights = &weights - &gradient * learning_rate;
            
            prev_cost = cost;
        }
        
        Ok(weights)
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_linear_regression_simple() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];
        
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();
        
        let predictions = model.predict(&x).unwrap();
        
        for (pred, actual) in predictions.iter().zip(y.iter()) {
            assert!((pred - actual).abs() < 1e-10);
        }
    }

    #[test]
    fn test_linear_regression_without_intercept() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];
        
        let mut model = LinearRegression::with_intercept(false);
        model.fit(&x, &y).unwrap();
        
        let coeffs = model.coefficients.as_ref().unwrap();
        assert!((coeffs[0] - 2.0).abs() < 1e-10);
        assert_eq!(model.intercept.unwrap(), 0.0);
    }

    #[test]
    fn test_linear_regression_multivariate() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![5.0, 8.0, 11.0, 14.0];
        
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();
        
        let predictions = model.predict(&x).unwrap();
        let score = model.score(&x, &y).unwrap();
        
        assert!(score > 0.95);
        for (pred, actual) in predictions.iter().zip(y.iter()) {
            assert!((pred - actual).abs() < 0.1);
        }
    }

    #[test]
    fn test_predict_without_fit() {
        let x = array![[1.0], [2.0]];
        let model = LinearRegression::new();
        
        assert!(model.predict(&x).is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0, 3.0];
        
        let mut model = LinearRegression::new();
        assert!(model.fit(&x, &y).is_err());
    }
}