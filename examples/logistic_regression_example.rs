use nametbd::{LogisticRegression, Matrix, Vector, Dataset, StandardScaler};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Logistic Regression Classification Example ===\n");
    
    // Create sample binary classification data
    // Feature: hours studied, Target: pass (1) or fail (0)
    let hours_studied = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
    let passed = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    
    println!("Training data:");
    println!("Hours studied: {:?}", hours_studied.column(0));
    println!("Passed exam: {:?}", passed);
    
    // Create dataset and split
    let dataset = Dataset::new(hours_studied.clone(), passed.clone())?;
    let (train_data, test_data) = dataset.train_test_split(0.25)?;
    
    println!("\nDataset split:");
    println!("Training samples: {}", train_data.n_samples());
    println!("Test samples: {}", test_data.n_samples());
    
    // Optional: Standardize features (though not critical for this simple example)
    let mut scaler = StandardScaler::new();
    let train_features_scaled = scaler.fit_transform(&train_data.features)?;
    let test_features_scaled = scaler.transform(&test_data.features)?;
    
    // Train logistic regression model
    let mut model = LogisticRegression::new();
    model.fit(&train_features_scaled, &train_data.labels)?;
    
    println!("\nModel training completed!");
    
    // Make predictions on test set
    let test_predictions = model.predict(&test_features_scaled)?;
    let test_probabilities = model.predict_proba(&test_features_scaled)?;
    
    // Evaluate model
    let accuracy = model.score(&test_features_scaled, &test_data.labels)?;
    
    println!("\nTest Results:");
    println!("Accuracy: {:.2}%", accuracy * 100.0);
    
    println!("\nPredictions vs Actual:");
    for (i, (&actual, (&predicted, &probability))) in test_data.labels.iter()
        .zip(test_predictions.iter().zip(test_probabilities.iter()))
        .enumerate() 
    {
        println!("Sample {}: Actual={:.0}, Predicted={:.0}, Probability={:.3}", 
                 i+1, actual, predicted, probability);
    }
    
    // Demonstrate on new data
    println!("\nPredictions on new data:");
    let new_hours = array![[2.5], [5.5], [9.0]];
    let new_hours_scaled = scaler.transform(&new_hours)?;
    let new_predictions = model.predict(&new_hours_scaled)?;
    let new_probabilities = model.predict_proba(&new_hours_scaled)?;
    
    for ((&hours, &prediction), &probability) in new_hours.column(0).iter()
        .zip(new_predictions.iter())
        .zip(new_probabilities.iter()) 
    {
        let outcome = if prediction == 1.0 { "Pass" } else { "Fail" };
        println!("Hours: {:.1} → Prediction: {} (probability: {:.3})", 
                 hours, outcome, probability);
    }
    
    // Calculate detailed metrics on full dataset for demonstration
    let full_predictions = model.predict(&scaler.transform(&hours_studied)?)?;
    
    let precision = nametbd::metrics::precision_score(&passed, &full_predictions)?;
    let recall = nametbd::metrics::recall_score(&passed, &full_predictions)?;
    let f1 = nametbd::metrics::f1_score(&passed, &full_predictions)?;
    let confusion_matrix = nametbd::metrics::confusion_matrix(&passed, &full_predictions)?;
    
    println!("\nDetailed Metrics on Full Dataset:");
    println!("Precision: {:.3}", precision);
    println!("Recall: {:.3}", recall);
    println!("F1-Score: {:.3}", f1);
    println!("Confusion Matrix:");
    println!("[[{:.0}, {:.0}]", confusion_matrix[(0,0)], confusion_matrix[(0,1)]);
    println!(" [{:.0}, {:.0}]]", confusion_matrix[(1,0)], confusion_matrix[(1,1)]);
    println!("(TN, FP)");
    println!("(FN, TP)");
    
    Ok(())
}