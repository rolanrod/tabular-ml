use crate::{Matrix, Vector};

#[derive(Clone, Debug)]
pub struct SVC {
    pub support_vectors: Option<Matrix>,
    pub dual_coefficients: Option<Vector>,
    pub intercept: Option<f64>,
    pub support_vector_indices: Option<Vec<usize>>,
    c: f64,
    kernel: Kernel,
    gamma: f64,
    degree: usize,
    coef0: f64,
    tolerance: f64,
    max_iter: usize,
    cache_size: f64,
}

#[derive(Clone, Debug)]
pub enum Kernel {
    Linear,
    Polynomial,
    Rbf,
    Sigmoid,
}

impl SVC {
    pub fn new() -> Self {
        Self {
            support_vectors: None,
            dual_coefficients: None,
            intercept: None,
            support_vector_indices: None,
            c: 1.0,
            kernel: Kernel::Rbf,
            gamma: 1.0,
            degree: 3,
            coef0: 0.0,
            tolerance: 1e-3,
            max_iter: -1_i32 as usize, // sklearn default: no limit
            cache_size: 200.0,
        }
    }

    pub fn c(mut self, c: f64) -> Self {
        if c <= 0.0 {
            panic!("C must be positive, got {}", c);
        }
        self.c = c;
        self
    }

    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn gamma(mut self, gamma: f64) -> Self {
        if gamma <= 0.0 {
            panic!("gamma must be positive, got {}", gamma);
        }
        self.gamma = gamma;
        self
    }

    pub fn degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }

    pub fn coef0(mut self, coef0: f64) -> Self {
        self.coef0 = coef0;
        self
    }

    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn fit(&mut self, x: &Matrix, y: &Vector) -> Result<(), String> {
        if x.nrows() != y.len() {
            return Err("Number of samples in X and y must match".to_string());
        }

        if x.nrows() == 0 {
            return Err("X must have at least one sample".to_string());
        }

        self.validate_labels(y)?;
        
        let (support_vectors, dual_coeffs, intercept, support_indices) = 
            self.smo_algorithm(x, y)?;
        
        self.support_vectors = Some(support_vectors);
        self.dual_coefficients = Some(dual_coeffs);
        self.intercept = Some(intercept);
        self.support_vector_indices = Some(support_indices);
        
        Ok(())
    }

    pub fn predict(&self, x: &Matrix) -> Result<Vector, String> {
        let decision_scores = self.decision_function(x)?;
        let predictions = decision_scores.mapv(|score| if score >= 0.0 { 1.0 } else { -1.0 });
        Ok(predictions)
    }

    pub fn decision_function(&self, x: &Matrix) -> Result<Vector, String> {
        let support_vectors = self.support_vectors.as_ref()
            .ok_or("Model not fitted. Call fit() first.")?;
        let dual_coeffs = self.dual_coefficients.as_ref()
            .ok_or("Model not fitted. Call fit() first.")?;
        let intercept = self.intercept.unwrap_or(0.0);

        if x.ncols() != support_vectors.ncols() {
            return Err(format!(
                "Number of features in X ({}) doesn't match training data ({})",
                x.ncols(),
                support_vectors.ncols()
            ));
        }

        let mut scores = Vector::zeros(x.nrows());
        
        for i in 0..x.nrows() {
            let mut score = 0.0;
            for j in 0..support_vectors.nrows() {
                let kernel_value = self.kernel_function(
                    &x.row(i).to_owned(),
                    &support_vectors.row(j).to_owned()
                );
                score += dual_coeffs[j] * kernel_value;
            }
            scores[i] = score + intercept;
        }
        
        Ok(scores)
    }

    pub fn score(&self, x: &Matrix, y: &Vector) -> Result<f64, String> {
        let predictions = self.predict(x)?;
        let accuracy = predictions.iter()
            .zip(y.iter())
            .map(|(pred, actual)| if (pred - actual).abs() < 1e-10 { 1.0 } else { 0.0 })
            .sum::<f64>() / y.len() as f64;
        Ok(accuracy)
    }

    fn validate_labels(&self, y: &Vector) -> Result<(), String> {
        let unique_labels: std::collections::HashSet<_> = y.iter()
            .map(|&val| (val * 1000.0).round() as i32)
            .collect();
        
        if unique_labels.len() != 2 {
            return Err("SVM requires exactly 2 classes".to_string());
        }
        
        let has_positive_one = unique_labels.contains(&1000);
        let has_negative_one = unique_labels.contains(&-1000);
        
        if !has_positive_one || !has_negative_one {
            return Err("Labels must be -1 and +1 for binary SVM".to_string());
        }
        
        Ok(())
    }

    fn kernel_function(&self, x1: &Vector, x2: &Vector) -> f64 {
        match self.kernel {
            Kernel::Linear => x1.dot(x2),
            Kernel::Polynomial => (self.gamma * x1.dot(x2) + self.coef0).powf(self.degree as f64),
            Kernel::Rbf => {
                let diff = x1 - x2;
                let norm_squared = diff.mapv(|x| x * x).sum();
                (-self.gamma * norm_squared).exp()
            },
            Kernel::Sigmoid => (self.gamma * x1.dot(x2) + self.coef0).tanh(),
        }
    }

    fn smo_algorithm(&self, x: &Matrix, y: &Vector) -> Result<(Matrix, Vector, f64, Vec<usize>), String> {
        let n_samples = x.nrows();
        let mut alphas = Vector::zeros(n_samples);
        let mut b = 0.0;
        
        let mut error_cache = Vector::zeros(n_samples);
        for i in 0..n_samples {
            error_cache[i] = -y[i];
        }
        
        let max_iterations = if self.max_iter == (-1_i32 as usize) { 
            1000 
        } else { 
            self.max_iter 
        };
        
        for _iter in 0..max_iterations {
            let mut num_changed = 0;
            
            for i in 0..n_samples {
                let error_i = error_cache[i];
                let r_i = error_i * y[i];
                
                if (r_i < -self.tolerance && alphas[i] < self.c) || 
                   (r_i > self.tolerance && alphas[i] > 0.0) {
                    
                    let j = self.select_second_alpha(i, &error_cache);
                    if let Some(j) = j {
                        if self.take_step(i, j, x, y, &mut alphas, &mut b, &mut error_cache)? {
                            num_changed += 1;
                        }
                    }
                }
            }
            
            if num_changed == 0 {
                break;
            }
        }
        
        let support_indices: Vec<usize> = alphas.iter()
            .enumerate()
            .filter(|&(_, &alpha)| alpha > 1e-8)
            .map(|(i, _)| i)
            .collect();
        
        if support_indices.is_empty() {
            return Err("No support vectors found".to_string());
        }
        
        let mut support_vectors = Matrix::zeros((support_indices.len(), x.ncols()));
        let mut dual_coeffs = Vector::zeros(support_indices.len());
        
        for (sv_idx, &original_idx) in support_indices.iter().enumerate() {
            support_vectors.row_mut(sv_idx).assign(&x.row(original_idx));
            dual_coeffs[sv_idx] = alphas[original_idx] * y[original_idx];
        }
        
        Ok((support_vectors, dual_coeffs, b, support_indices))
    }

    fn select_second_alpha(&self, i: usize, error_cache: &Vector) -> Option<usize> {
        let error_i = error_cache[i];
        let mut best_j = None;
        let mut max_step = 0.0;
        
        for j in 0..error_cache.len() {
            if j != i {
                let error_j = error_cache[j];
                let step = (error_i - error_j).abs();
                if step > max_step {
                    max_step = step;
                    best_j = Some(j);
                }
            }
        }
        
        best_j
    }

    fn take_step(
        &self,
        i: usize,
        j: usize,
        x: &Matrix,
        y: &Vector,
        alphas: &mut Vector,
        b: &mut f64,
        error_cache: &mut Vector,
    ) -> Result<bool, String> {
        if i == j {
            return Ok(false);
        }
        
        let alpha_i_old = alphas[i];
        let alpha_j_old = alphas[j];
        let y_i = y[i];
        let y_j = y[j];
        
        let (l, h) = if y_i != y_j {
            let l = (0.0_f64).max(alpha_j_old - alpha_i_old);
            let h = self.c.min(self.c + alpha_j_old - alpha_i_old);
            (l, h)
        } else {
            let l = (0.0_f64).max(alpha_i_old + alpha_j_old - self.c);
            let h = self.c.min(alpha_i_old + alpha_j_old);
            (l, h)
        };
        
        if (l - h).abs() < 1e-8 {
            return Ok(false);
        }
        
        let k_ii = self.kernel_function(&x.row(i).to_owned(), &x.row(i).to_owned());
        let k_ij = self.kernel_function(&x.row(i).to_owned(), &x.row(j).to_owned());
        let k_jj = self.kernel_function(&x.row(j).to_owned(), &x.row(j).to_owned());
        
        let eta = k_ii + k_jj - 2.0 * k_ij;
        
        let alpha_j_new = if eta > 0.0 {
            let alpha_j_unc = alpha_j_old + y_j * (error_cache[i] - error_cache[j]) / eta;
            if alpha_j_unc >= h {
                h
            } else if alpha_j_unc <= l {
                l
            } else {
                alpha_j_unc
            }
        } else {
            return Ok(false);
        };
        
        if (alpha_j_new - alpha_j_old).abs() < 1e-8 {
            return Ok(false);
        }
        
        let alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new);
        
        let b1 = error_cache[i] + y_i * (alpha_i_new - alpha_i_old) * k_ii +
                 y_j * (alpha_j_new - alpha_j_old) * k_ij + *b;
        let b2 = error_cache[j] + y_i * (alpha_i_new - alpha_i_old) * k_ij +
                 y_j * (alpha_j_new - alpha_j_old) * k_jj + *b;
        
        let b_new = if alpha_i_new > 0.0 && alpha_i_new < self.c {
            b1
        } else if alpha_j_new > 0.0 && alpha_j_new < self.c {
            b2
        } else {
            (b1 + b2) / 2.0
        };
        
        alphas[i] = alpha_i_new;
        alphas[j] = alpha_j_new;
        *b = b_new;
        
        for k in 0..x.nrows() {
            if k != i && k != j {
                let k_ki = self.kernel_function(&x.row(k).to_owned(), &x.row(i).to_owned());
                let k_kj = self.kernel_function(&x.row(k).to_owned(), &x.row(j).to_owned());
                error_cache[k] += y_i * (alpha_i_new - alpha_i_old) * k_ki +
                                  y_j * (alpha_j_new - alpha_j_old) * k_kj + 
                                  (*b - b_new);
            }
        }
        
        error_cache[i] = 0.0;
        error_cache[j] = 0.0;
        
        Ok(true)
    }

    pub fn n_support_vectors(&self) -> Option<usize> {
        self.support_vectors.as_ref().map(|sv| sv.nrows())
    }
}

impl Default for SVC {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_svc_linear_separable() {
        let x = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [-1.0, -1.0],
            [-2.0, -2.0],
            [-3.0, -3.0]
        ];
        let y = array![1.0, 1.0, 1.0, -1.0, -1.0, -1.0];
        
        let mut svm = SVC::new().kernel(Kernel::Linear).c(1.0);
        svm.fit(&x, &y).unwrap();
        
        let predictions = svm.predict(&x).unwrap();
        let accuracy = svm.score(&x, &y).unwrap();
        
        assert!(accuracy >= 0.8);
        assert!(svm.n_support_vectors().unwrap() >= 2);
    }

    #[test]
    fn test_svc_rbf_kernel() {
        let x = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [5.0, 5.0],
            [6.0, 5.0],
            [5.0, 6.0]
        ];
        let y = array![1.0, 1.0, 1.0, -1.0, -1.0, -1.0];
        
        let mut svm = SVC::new().kernel(Kernel::Rbf).gamma(1.0).c(1.0);
        svm.fit(&x, &y).unwrap();
        
        let accuracy = svm.score(&x, &y).unwrap();
        assert!(accuracy >= 0.8);
    }

    #[test]
    fn test_svc_polynomial_kernel() {
        let x = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [-1.0, -1.0],
            [-2.0, -2.0]
        ];
        let y = array![1.0, 1.0, -1.0, -1.0];
        
        let mut svm = SVC::new().kernel(Kernel::Polynomial).degree(2).c(1.0);
        svm.fit(&x, &y).unwrap();
        
        let accuracy = svm.score(&x, &y).unwrap();
        assert!(accuracy >= 0.5);
    }

    #[test]
    fn test_svc_invalid_labels() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 2.0];
        
        let mut svm = SVC::new();
        assert!(svm.fit(&x, &y).is_err());
    }

    #[test]
    fn test_svc_wrong_number_classes() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![1.0, 1.0, 1.0];
        
        let mut svm = SVC::new();
        assert!(svm.fit(&x, &y).is_err());
    }

    #[test]
    fn test_svc_predict_without_fit() {
        let x = array![[1.0], [2.0]];
        let svm = SVC::new();
        
        assert!(svm.predict(&x).is_err());
        assert!(svm.decision_function(&x).is_err());
    }

    #[test]
    fn test_svc_dimension_mismatch() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, -1.0, 1.0];
        
        let mut svm = SVC::new();
        assert!(svm.fit(&x, &y).is_err());
    }

    #[test]
    fn test_kernel_functions() {
        let svm = SVC::new().kernel(Kernel::Linear);
        let x1 = array![1.0, 2.0];
        let x2 = array![3.0, 4.0];
        
        let linear = svm.kernel_function(&x1, &x2);
        assert!((linear - 11.0).abs() < 1e-10);
        
        let rbf_svm = SVC::new().kernel(Kernel::Rbf).gamma(1.0);
        let rbf = rbf_svm.kernel_function(&x1, &x2);
        assert!(rbf > 0.0 && rbf <= 1.0);
    }

    #[test]
    fn test_svc_invalid_c() {
        std::panic::catch_unwind(|| {
            SVC::new().c(-1.0);
        }).expect_err("Should panic on negative C");
    }

    #[test]
    fn test_svc_invalid_gamma() {
        std::panic::catch_unwind(|| {
            SVC::new().gamma(-1.0);
        }).expect_err("Should panic on negative gamma");
    }
}