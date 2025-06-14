use crate::{Matrix, Vector};
use ndarray::{Array2, concatenate, Axis, s};

pub struct LinearRegression {
    W: Option<Vector>,
    b: Option<f64
}

impl LinearRegression {
    pub fn new() -> Self {
        Self { coefficients: None, intercept: None }
    }

    pub fn fit(&mut self, X: &Matrix, y: &Vector) -> Result<(), String> {
        let X_b = concatenate![Axis(1), Array2::ones((X.nrows(), 1)), X.view()];
        let W = (X_b.t().dot(&X_b).inv()).dot(&X_b.t()).dot(&y);
        self.W = Some(W);

        Ok(())
    }

    pub fn predict(&self, X: &Matrix) -> Result<Vector, String> {
        let X_b = concatenate![Axis(1), Array2::ones((X.nrows(), 1)), X.view()];
        let pred = X_b.dot(&self.W.as_ref().ok_or("Model not fitted. Call fit() first.")?);

        Ok(pred)
    }
}


pub struct SGDRegressor {
    
}

// use crate::{Matrix, Vector};
// use ndarray::{Array2, concatenate, Axis, s};

// pub struct LinearRegression {
//     coefficients: Option<Vector>,
//     intercept: Option<f64>,
// }

// impl LinearRegression {
//     pub fn new() -> Self {
//         Self {
//             coefficients: None,
//             intercept: None,
//         }
//     }
    
//     pub fn fit(&mut self, x: &Matrix, y: &Vector) -> Result<(), String> {
//         if x.nrows() != y.len() {
//             return Err("Number of samples in X and y must match".to_string());
//         }
        
//         // Add intercept column (column of ones)
//         let ones = Array2::ones((x.nrows(), 1));
//         let x_with_intercept = concatenate![Axis(1), ones, x.view()];
        
//         // Solve normal equation: (X^T * X)^-1 * X^T * y
//         let xt = x_with_intercept.t();
//         let xtx = xt.dot(&x_with_intercept);
//         let xty = xt.dot(y);
        
//         // For now, use a simple solver (in practice we'd want something more robust)
//         match self.solve_linear_system(&xtx, &xty) {
//             Some(params) => {
//                 self.intercept = Some(params[0]);
//                 self.coefficients = Some(params.slice(s![1..]).to_owned());
//                 Ok(())
//             }
//             None => Err("Failed to solve linear system".to_string()),
//         }
//     }
    
//     pub fn predict(&self, x: &Matrix) -> Result<Vector, String> {
//         let coefficients = self.coefficients.as_ref()
//             .ok_or("Model not fitted. Call fit() first.")?;
//         let intercept = self.intercept
//             .ok_or("Model not fitted. Call fit() first.")?;
        
//         let predictions = x.dot(coefficients) + intercept;
//         Ok(predictions)
//     }
    
//     // Simple Gaussian elimination (not numerically stable for production use)
//     fn solve_linear_system(&self, a: &Matrix, b: &Vector) -> Option<Vector> {
//         let n = a.nrows();
//         if n != a.ncols() || n != b.len() {
//             return None;
//         }
        
//         let mut aug = Matrix::zeros((n, n + 1));
//         aug.slice_mut(s![.., ..n]).assign(a);
//         aug.column_mut(n).assign(b);
        
//         // Forward elimination
//         for i in 0..n {
//             let pivot = aug[[i, i]];
//             if pivot.abs() < 1e-10 {
//                 return None; // Singular matrix
//             }
            
//             for j in (i + 1)..n {
//                 let factor = aug[[j, i]] / pivot;
//                 for k in 0..=n {
//                     aug[[j, k]] -= factor * aug[[i, k]];
//                 }
//             }
//         }
        
//         // Back substitution
//         let mut x = Vector::zeros(n);
//         for i in (0..n).rev() {
//             let mut sum = 0.0;
//             for j in (i + 1)..n {
//                 sum += aug[[i, j]] * x[j];
//             }
//             x[i] = (aug[[i, n]] - sum) / aug[[i, i]];
//         }
        
//         Some(x)
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_linear_regression() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0]; // y = 2x
        
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();
        
        let predictions = model.predict(&x).unwrap();
        for (pred, actual) in predictions.iter().zip(y.iter()) {
            assert!((pred - actual).abs() < 1e-10);
        }
    }
}