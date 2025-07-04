use nametbd::{SVC, Kernel, Dataset, StandardScaler, Matrix, Vector};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Support Vector Machine (SVM) Example ===\n");
    
    // Create sample classification data
    let x = array![
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
        [-1.0, -1.0],
        [-2.0, -2.0],
        [-3.0, -3.0],
        [-4.0, -4.0],
        [2.0, -2.0],
        [-2.0, 2.0]
    ];
    
    // Binary classification: +1 for positive quadrant bias, -1 for negative
    let y = array![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0];
    
    println!("Training data: {} samples, {} features", x.nrows(), x.ncols());
    println!("Classes: +1 and -1\n");
    
    // Create dataset and split
    let dataset = Dataset::new(x, y)?;
    let (train_data, test_data) = dataset.train_test_split(0.3)?;
    
    // Standardize features for better SVM performance
    let mut scaler = StandardScaler::new();
    let train_features_scaled = scaler.fit_transform(&train_data.features)?;
    let test_features_scaled = scaler.transform(&test_data.features)?;
    
    println!("Training samples: {}, Test samples: {}\n", 
             train_data.n_samples(), test_data.n_samples());
    
    // Test different SVM configurations
    let svm_configs = vec![
        ("Linear SVM", SVC::new().kernel(Kernel::Linear).c(1.0)),
        ("RBF SVM (γ=1.0)", SVC::new().kernel(Kernel::Rbf).gamma(1.0).c(1.0)),
        ("RBF SVM (γ=0.1)", SVC::new().kernel(Kernel::Rbf).gamma(0.1).c(1.0)),
        ("Polynomial SVM (degree=2)", SVC::new().kernel(Kernel::Polynomial).degree(2).c(1.0)),
        ("Polynomial SVM (degree=3)", SVC::new().kernel(Kernel::Polynomial).degree(3).c(1.0)),
        ("Sigmoid SVM", SVC::new().kernel(Kernel::Sigmoid).c(1.0)),
    ];
    
    println!("SVM Configuration Comparison:");
    println!("{:<25} {:>12} {:>12} {:>15}", "Model", "Train Acc", "Test Acc", "Support Vectors");
    println!("{}", "-".repeat(70));
    
    for (name, mut svm) in svm_configs {
        // Train the SVM
        svm.fit(&train_features_scaled, &train_data.labels)?;
        
        // Evaluate
        let train_score = svm.score(&train_features_scaled, &train_data.labels)?;
        let test_score = svm.score(&test_features_scaled, &test_data.labels)?;
        let n_support = svm.n_support_vectors().unwrap_or(0);
        
        println!("{:<25} {:>12.4} {:>12.4} {:>15}", 
                 name, train_score, test_score, n_support);
    }
    
    println!("\n=== Detailed RBF SVM Analysis ===");
    
    // Focus on RBF SVM with different C values
    let c_values = vec![0.1, 1.0, 10.0, 100.0];
    
    println!("\nEffect of C parameter on RBF SVM:");
    println!("{:<10} {:>12} {:>12} {:>15}", "C", "Train Acc", "Test Acc", "Support Vectors");
    println!("{}", "-".repeat(55));
    
    for &c in &c_values {
        let mut svm = SVC::new().kernel(Kernel::Rbf).gamma(1.0).c(c);
        svm.fit(&train_features_scaled, &train_data.labels)?;
        
        let train_score = svm.score(&train_features_scaled, &train_data.labels)?;
        let test_score = svm.score(&test_features_scaled, &test_data.labels)?;
        let n_support = svm.n_support_vectors().unwrap_or(0);
        
        println!("{:<10.1} {:>12.4} {:>12.4} {:>15}", 
                 c, train_score, test_score, n_support);
    }
    
    // Demonstrate decision function
    println!("\n=== Decision Function Demo ===");
    let mut demo_svm = SVC::new().kernel(Kernel::Rbf).gamma(1.0).c(1.0);
    demo_svm.fit(&train_features_scaled, &train_data.labels)?;
    
    let decision_scores = demo_svm.decision_function(&test_features_scaled)?;
    let predictions = demo_svm.predict(&test_features_scaled)?;
    
    println!("Sample predictions vs decision scores:");
    println!("{:<12} {:>15} {:>12}", "True Label", "Decision Score", "Prediction");
    println!("{}", "-".repeat(40));
    
    for i in 0..test_data.labels.len().min(5) {
        println!("{:<12.0} {:>15.4} {:>12.0}", 
                 test_data.labels[i], decision_scores[i], predictions[i]);
    }
    
    println!("\nKey Observations:");
    println!("• Linear SVM works well for linearly separable data");
    println!("• RBF kernel can capture non-linear relationships");
    println!("• Higher C values reduce regularization (may overfit)");
    println!("• Lower C values increase regularization (may underfit)");
    println!("• Decision scores indicate confidence: |score| = distance from boundary");
    println!("• More support vectors often indicate more complex decision boundary");
    
    Ok(())
}