use crate::{Matrix, Vector};
use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct LDA {
    pub components: Option<Matrix>,
    pub explained_variance_ratio: Option<Vector>,
    pub means: Option<Matrix>,
    pub classes: Option<Vector>,
    n_components: Option<usize>,
    solver: String,
}

impl LDA {
    pub fn new() -> Self {
        Self {
            components: None,
            explained_variance_ratio: None,
            means: None,
            classes: None,
            n_components: None,
            solver: "svd".to_string(),
        }
    }

    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = Some(n_components);
        self
    }

    pub fn solver(mut self, solver: &str) -> Self {
        match solver {
            "svd" | "lsqr" | "eigen" => {
                self.solver = solver.to_string();
            }
            _ => panic!("Invalid solver: {}. Must be one of: svd, lsqr, eigen", solver),
        }
        self
    }

    pub fn fit(&mut self, x: &Matrix, y: &Vector) -> Result<(), String> {
        if x.nrows() != y.len() {
            return Err("Number of samples in X and y must match".to_string());
        }

        if x.nrows() == 0 || x.ncols() == 0 {
            return Err("Input matrix must have at least one sample and one feature".to_string());
        }

        // Get unique classes
        let mut unique_classes: Vec<f64> = y.iter().cloned().collect();
        unique_classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        unique_classes.dedup();

        if unique_classes.len() < 2 {
            return Err("LDA requires at least 2 classes".to_string());
        }

        let n_classes = unique_classes.len();
        let n_features = x.ncols();

        // Determine number of components (at most n_classes - 1)
        let max_components = n_classes - 1;
        let n_components = self.n_components.unwrap_or(max_components).min(max_components);

        if n_components == 0 {
            return Err("Number of components must be positive".to_string());
        }

        // Compute class means and overall mean
        let overall_mean = x.mean_axis(ndarray::Axis(0)).unwrap();
        let mut class_means = Matrix::zeros((n_classes, n_features));
        let mut class_counts = vec![0; n_classes];

        for (i, &class_label) in y.iter().enumerate() {
            let class_idx = unique_classes.iter().position(|&c| c == class_label).unwrap();
            class_counts[class_idx] += 1;
            for j in 0..n_features {
                class_means[[class_idx, j]] += x[[i, j]];
            }
        }

        // Normalize class means
        for i in 0..n_classes {
            if class_counts[i] > 0 {
                for j in 0..n_features {
                    class_means[[i, j]] /= class_counts[i] as f64;
                }
            }
        }

        // Compute within-class scatter matrix (Sw)
        let mut sw = Matrix::zeros((n_features, n_features));
        for (i, &class_label) in y.iter().enumerate() {
            let class_idx = unique_classes.iter().position(|&c| c == class_label).unwrap();
            let diff = x.row(i).to_owned() - class_means.row(class_idx);
            let outer_product = diff.view().insert_axis(ndarray::Axis(1))
                .dot(&diff.view().insert_axis(ndarray::Axis(0)));
            sw = sw + outer_product;
        }

        // Compute between-class scatter matrix (Sb)
        let mut sb = Matrix::zeros((n_features, n_features));
        for (class_idx, &class_count) in class_counts.iter().enumerate() {
            if class_count > 0 {
                let diff = class_means.row(class_idx).to_owned() - &overall_mean;
                let outer_product = diff.view().insert_axis(ndarray::Axis(1))
                    .dot(&diff.view().insert_axis(ndarray::Axis(0)));
                sb = sb + outer_product * (class_count as f64);
            }
        }

        // Solve generalized eigenvalue problem: Sb * v = λ * Sw * v
        let (eigenvalues, eigenvectors) = self.solve_generalized_eigenvalue(&sb, &sw)?;

        // Sort eigenvalues and eigenvectors in descending order
        let mut eigen_pairs: Vec<(f64, Vector)> = eigenvalues.iter()
            .zip(eigenvectors.axis_iter(ndarray::Axis(1)))
            .map(|(&val, vec)| (val, vec.to_owned()))
            .collect();
            
        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        // Filter out complex eigenvalues (negative or very small)
        eigen_pairs.retain(|(eigenval, _)| *eigenval > 1e-10);

        // Select top n_components
        let n_components = n_components.min(eigen_pairs.len());
        if n_components == 0 {
            return Err("No valid discriminant components found".to_string());
        }

        let mut components = Matrix::zeros((n_components, n_features));
        let mut selected_eigenvalues = Vector::zeros(n_components);

        for (i, (eigenval, eigenvec)) in eigen_pairs.iter().take(n_components).enumerate() {
            components.row_mut(i).assign(eigenvec);
            selected_eigenvalues[i] = *eigenval;
        }

        // Compute explained variance ratio
        let total_eigenvalue_sum: f64 = eigen_pairs.iter().map(|(val, _)| *val).sum();
        let explained_variance_ratio = if total_eigenvalue_sum > 0.0 {
            &selected_eigenvalues / total_eigenvalue_sum
        } else {
            Vector::zeros(n_components)
        };

        self.components = Some(components);
        self.explained_variance_ratio = Some(explained_variance_ratio);
        self.means = Some(class_means);
        self.classes = Some(unique_classes.into());

        Ok(())
    }

    pub fn transform(&self, x: &Matrix) -> Result<Matrix, String> {
        let components = self.components.as_ref()
            .ok_or("LDA not fitted. Call fit() first.")?;

        if x.ncols() != components.ncols() {
            return Err(format!(
                "Number of features in X ({}) doesn't match training data ({})",
                x.ncols(), components.ncols()
            ));
        }

        // Project data onto discriminant directions
        let transformed = x.dot(&components.t());
        Ok(transformed)
    }

    pub fn fit_transform(&mut self, x: &Matrix, y: &Vector) -> Result<Matrix, String> {
        self.fit(x, y)?;
        self.transform(x)
    }

    pub fn predict(&self, x: &Matrix) -> Result<Vector, String> {
        let components = self.components.as_ref()
            .ok_or("LDA not fitted. Call fit() first.")?;
        let class_means = self.means.as_ref()
            .ok_or("LDA not fitted. Call fit() first.")?;
        let classes = self.classes.as_ref()
            .ok_or("LDA not fitted. Call fit() first.")?;

        // Transform input data
        let x_transformed = self.transform(x)?;
        
        // Transform class means
        let class_means_transformed = class_means.dot(&components.t());
        
        let mut predictions = Vector::zeros(x.nrows());
        
        // Classify based on nearest class mean in transformed space
        for i in 0..x.nrows() {
            let mut min_distance = f64::INFINITY;
            let mut predicted_class = classes[0];
            
            for j in 0..class_means_transformed.nrows() {
                let diff = x_transformed.row(i).to_owned() - class_means_transformed.row(j);
                let distance = diff.mapv(|x| x * x).sum();
                
                if distance < min_distance {
                    min_distance = distance;
                    predicted_class = classes[j];
                }
            }
            
            predictions[i] = predicted_class;
        }
        
        Ok(predictions)
    }

    pub fn score(&self, x: &Matrix, y: &Vector) -> Result<f64, String> {
        let predictions = self.predict(x)?;
        let accuracy = predictions.iter()
            .zip(y.iter())
            .map(|(pred, actual)| if (pred - actual).abs() < 1e-10 { 1.0 } else { 0.0 })
            .sum::<f64>() / y.len() as f64;
        Ok(accuracy)
    }

    fn solve_generalized_eigenvalue(&self, sb: &Matrix, sw: &Matrix) -> Result<(Vector, Matrix), String> {
        // Simplified approach: use regularized Sw^(-1) * Sb
        // In practice, you'd use a proper generalized eigenvalue solver
        
        let n = sb.nrows();
        
        // Add regularization to Sw to ensure invertibility
        let reg_sw = sw + &Matrix::eye(n) * 1e-6;
        
        // Compute pseudo-inverse of Sw using SVD
        let sw_inv = self.pseudo_inverse(&reg_sw)?;
        
        // Compute Sw^(-1) * Sb
        let matrix = sw_inv.dot(sb);
        
        // Standard eigenvalue decomposition
        self.eigen_decomposition(&matrix)
    }

    fn pseudo_inverse(&self, matrix: &Matrix) -> Result<Matrix, String> {
        // Simplified pseudo-inverse using regularization
        // In practice, you'd use SVD-based pseudo-inverse
        let n = matrix.nrows();
        let reg_matrix = matrix + &Matrix::eye(n) * 1e-6;
        
        // Use iterative method for matrix inversion
        let mut inv = Matrix::eye(n);
        let mut current = reg_matrix.clone();
        
        for _ in 0..10 {
            let residual = &Matrix::eye(n) - &current.dot(&inv);
            inv = &inv + &inv.dot(&residual);
            current = &current + &residual.dot(&current);
            
            // Check convergence
            let error = residual.mapv(|x| x.abs()).into_iter().fold(0.0, f64::max);
            if error < 1e-10 {
                break;
            }
        }
        
        Ok(inv)
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
                lambda = v.dot(&a.dot(&v));
                
                if lambda.abs() < 1e-10 {
                    break;
                }
            }
            
            eigenvalues[i] = lambda;
            eigenvectors.column_mut(i).assign(&v);
            
            // Deflation
            if lambda.abs() > 1e-10 {
                let vv = v.view().insert_axis(ndarray::Axis(1)).dot(&v.view().insert_axis(ndarray::Axis(0)));
                a = &a - &(vv * lambda);
            }
        }
        
        Ok((eigenvalues, eigenvectors))
    }
}

impl Default for LDA {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_lda_basic() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [8.0, 9.0],
            [9.0, 10.0],
            [10.0, 11.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut lda = LDA::new();
        let transformed = lda.fit_transform(&x, &y).unwrap();
        
        assert_eq!(transformed.shape(), &[6, 1]); // 2 classes -> 1 component
        assert!(lda.components.is_some());
        assert!(lda.explained_variance_ratio.is_some());
    }

    #[test]
    fn test_lda_prediction() {
        let x = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [8.0, 8.0],
            [9.0, 9.0]
        ];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let mut lda = LDA::new();
        lda.fit(&x, &y).unwrap();
        
        let predictions = lda.predict(&x).unwrap();
        let accuracy = lda.score(&x, &y).unwrap();
        
        assert!(accuracy > 0.5);
    }

    #[test]
    fn test_lda_three_classes() {
        let x = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [5.0, 1.0],
            [6.0, 2.0],
            [1.0, 5.0],
            [2.0, 6.0]
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

        let mut lda = LDA::new();
        let transformed = lda.fit_transform(&x, &y).unwrap();
        
        assert_eq!(transformed.shape(), &[6, 2]); // 3 classes -> 2 components
    }

    #[test]
    fn test_lda_insufficient_classes() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 0.0]; // All same class
        
        let mut lda = LDA::new();
        assert!(lda.fit(&x, &y).is_err());
    }

    #[test]
    fn test_lda_dimension_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0]; // Different lengths
        
        let mut lda = LDA::new();
        assert!(lda.fit(&x, &y).is_err());
    }

    #[test]
    fn test_lda_transform_without_fit() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let lda = LDA::new();
        
        assert!(lda.transform(&x).is_err());
    }

    #[test]
    fn test_lda_components_limit() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [8.0, 9.0, 10.0],
            [9.0, 10.0, 11.0]
        ];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let mut lda = LDA::new().n_components(5); // More than n_classes - 1
        lda.fit(&x, &y).unwrap();
        
        // Should be limited to n_classes - 1 = 1
        assert_eq!(lda.components.as_ref().unwrap().nrows(), 1);
    }
}