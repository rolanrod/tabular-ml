use nametbd::{LinearRegression, Ridge, Lasso, ElasticNet, Matrix, Vector, Dataset, StandardScaler};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Regularized Linear Regression Comparison ===\n");
    
    // Create sample data with some noise and irrelevant features
    // Target: y = 3*x1 + 2*x2 + noise, x3 and x4 are irrelevant
    let x = array![
        [1.0, 2.0, 0.5, -0.3],
        [2.0, 3.0, -0.2, 0.8],
        [3.0, 1.0, 1.1, -0.5],
        [4.0, 4.0, 0.3, 0.2],
        [5.0, 2.0, -0.8, 0.7],
        [6.0, 5.0, 0.9, -0.1],
        [7.0, 3.0, -0.4, 0.6],
        [8.0, 6.0, 0.7, -0.9],
        [9.0, 4.0, -0.1, 0.4],
        [10.0, 7.0, 0.2, -0.2]
    ];
    
    // y = 3*x1 + 2*x2 + small noise
    let y = array![7.1, 11.9, 12.8, 19.7, 18.9, 27.8, 26.7, 35.9, 34.8, 43.1];
    
    println!("Training data shape: {} samples, {} features", x.nrows(), x.ncols());
    println!("True relationship: y = 3*x1 + 2*x2 + noise (x3, x4 are irrelevant)\n");
    
    // Create dataset and split
    let dataset = Dataset::new(x, y)?;
    let (train_data, test_data) = dataset.train_test_split(0.3)?;
    
    // Standardize features
    let mut scaler = StandardScaler::new();
    let train_features_scaled = scaler.fit_transform(&train_data.features)?;
    let test_features_scaled = scaler.transform(&test_data.features)?;
    
    println!("Training samples: {}, Test samples: {}\n", 
             train_data.n_samples(), test_data.n_samples());
    
    // Compare different regression methods
    let models: Vec<(&str, Box<dyn RegressionModel>)> = vec![
        ("Linear Regression", Box::new(LinearRegressionWrapper::new())),
        ("Ridge (α=0.1)", Box::new(RidgeWrapper::new(0.1))),
        ("Ridge (α=1.0)", Box::new(RidgeWrapper::new(1.0))),
        ("Lasso (α=0.1)", Box::new(LassoWrapper::new(0.1))),
        ("Lasso (α=0.5)", Box::new(LassoWrapper::new(0.5))),
        ("ElasticNet (α=0.1, l1=0.5)", Box::new(ElasticNetWrapper::new(0.1, 0.5))),
        ("ElasticNet (α=0.5, l1=0.7)", Box::new(ElasticNetWrapper::new(0.5, 0.7))),
    ];
    
    println!("Model Comparison Results:");
    println!("{:<25} {:>10} {:>10} {:>15} {:>15} {:>15} {:>15}", 
             "Model", "Train R²", "Test R²", "Coef 1", "Coef 2", "Coef 3", "Coef 4");
    println!("{}", "-".repeat(100));
    
    for (name, mut model) in models {
        // Train model
        model.fit(&train_features_scaled, &train_data.labels)?;
        
        // Evaluate
        let train_score = model.score(&train_features_scaled, &train_data.labels)?;
        let test_score = model.score(&test_features_scaled, &test_data.labels)?;
        let coeffs = model.coefficients();
        
        println!("{:<25} {:>10.4} {:>10.4} {:>15.4} {:>15.4} {:>15.4} {:>15.4}", 
                 name, train_score, test_score, 
                 coeffs[0], coeffs[1], coeffs[2], coeffs[3]);
    }
    
    println!("\nObservations:");
    println!("• Linear regression may overfit with noise");
    println!("• Ridge shrinks all coefficients but keeps all features");
    println!("• Lasso can drive irrelevant coefficients to zero (feature selection)");
    println!("• ElasticNet combines benefits of both Ridge and Lasso");
    println!("• Higher α values increase regularization strength");
    
    // Demonstrate regularization path for Lasso
    println!("\n=== Lasso Regularization Path ===");
    let alphas = vec![0.001, 0.01, 0.1, 0.5, 1.0, 5.0];
    
    println!("{:<10} {:>10} {:>15} {:>15} {:>15} {:>15}", 
             "Alpha", "Test R²", "Coef 1", "Coef 2", "Coef 3", "Coef 4");
    println!("{}", "-".repeat(75));
    
    for &alpha in &alphas {
        let mut lasso = Lasso::new().alpha(alpha);
        lasso.fit(&train_features_scaled, &train_data.labels)?;
        
        let test_score = lasso.score(&test_features_scaled, &test_data.labels)?;
        let coeffs = lasso.coefficients.as_ref().unwrap();
        
        println!("{:<10.3} {:>10.4} {:>15.4} {:>15.4} {:>15.4} {:>15.4}", 
                 alpha, test_score, coeffs[0], coeffs[1], coeffs[2], coeffs[3]);
    }
    
    println!("\nNotice how Lasso drives irrelevant coefficients (3 & 4) to zero as α increases!");
    
    Ok(())
}

// Trait to unify different regression models for comparison
trait RegressionModel {
    fn fit(&mut self, x: &Matrix, y: &Vector) -> Result<(), String>;
    fn score(&self, x: &Matrix, y: &Vector) -> Result<f64, String>;
    fn coefficients(&self) -> Vector;
}

// Wrapper structs to implement the trait
struct LinearRegressionWrapper {
    model: LinearRegression,
}

impl LinearRegressionWrapper {
    fn new() -> Self {
        Self { model: LinearRegression::new() }
    }
}

impl RegressionModel for LinearRegressionWrapper {
    fn fit(&mut self, x: &Matrix, y: &Vector) -> Result<(), String> {
        self.model.fit(x, y)
    }
    
    fn score(&self, x: &Matrix, y: &Vector) -> Result<f64, String> {
        self.model.score(x, y)
    }
    
    fn coefficients(&self) -> Vector {
        self.model.coefficients.as_ref().unwrap().clone()
    }
}

struct RidgeWrapper {
    model: Ridge,
}

impl RidgeWrapper {
    fn new(alpha: f64) -> Self {
        Self { model: Ridge::new().alpha(alpha) }
    }
}

impl RegressionModel for RidgeWrapper {
    fn fit(&mut self, x: &Matrix, y: &Vector) -> Result<(), String> {
        self.model.fit(x, y)
    }
    
    fn score(&self, x: &Matrix, y: &Vector) -> Result<f64, String> {
        self.model.score(x, y)
    }
    
    fn coefficients(&self) -> Vector {
        self.model.coefficients.as_ref().unwrap().clone()
    }
}

struct LassoWrapper {
    model: Lasso,
}

impl LassoWrapper {
    fn new(alpha: f64) -> Self {
        Self { model: Lasso::new().alpha(alpha) }
    }
}

impl RegressionModel for LassoWrapper {
    fn fit(&mut self, x: &Matrix, y: &Vector) -> Result<(), String> {
        self.model.fit(x, y)
    }
    
    fn score(&self, x: &Matrix, y: &Vector) -> Result<f64, String> {
        self.model.score(x, y)
    }
    
    fn coefficients(&self) -> Vector {
        self.model.coefficients.as_ref().unwrap().clone()
    }
}

struct ElasticNetWrapper {
    model: ElasticNet,
}

impl ElasticNetWrapper {
    fn new(alpha: f64, l1_ratio: f64) -> Self {
        Self { model: ElasticNet::new().alpha(alpha).l1_ratio(l1_ratio) }
    }
}

impl RegressionModel for ElasticNetWrapper {
    fn fit(&mut self, x: &Matrix, y: &Vector) -> Result<(), String> {
        self.model.fit(x, y)
    }
    
    fn score(&self, x: &Matrix, y: &Vector) -> Result<f64, String> {
        self.model.score(x, y)
    }
    
    fn coefficients(&self) -> Vector {
        self.model.coefficients.as_ref().unwrap().clone()
    }
}