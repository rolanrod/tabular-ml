use nametbd::{LinearRegression, Matrix, Vector};
use ndarray::array;

fn main() -> Result<(), String> {
    // Simple example with synthetic data
    println!("=== Simple Linear Regression Example ===\n");
    
    // Create sample data: y = 2x + 3 + noise
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![5.1, 6.9, 9.2, 11.1, 12.8]; // 2x + 3 with small noise
    
    println!("Training data:");
    println!("X: {:?}", x);
    println!("y: {:?}", y);
    
    // Train model
    let mut model = LinearRegression::new();
    model.fit(&x, &y)?;
    
    // Make predictions
    let predictions = model.predict(&x)?;
    
    // Evaluate
    let score = model.score(&x, &y)?;
    let mse = nametbd::metrics::mean_squared_error(&y, &predictions)?;
    
    println!("\nResults:");
    println!("Coefficient: {:.4}", model.coefficients.as_ref().unwrap()[0]);
    println!("Intercept: {:.4}", model.intercept.unwrap());
    println!("R² score: {:.4}", score);
    println!("MSE: {:.4}", mse);
    
    println!("\nPredictions vs Actual:");
    for (i, (pred, actual)) in predictions.iter().zip(y.iter()).enumerate() {
        println!("Sample {}: Predicted={:.2}, Actual={:.2}, Error={:.2}", 
                 i+1, pred, actual, (pred - actual).abs());
    }
    
    // Test on new data
    let new_x = array![[6.0], [7.0]];
    let new_predictions = model.predict(&new_x)?;
    println!("\nPredictions on new data:");
    for (i, pred) in new_predictions.iter().enumerate() {
        println!("X={:.1}: Predicted y={:.2}", new_x[(i, 0)], pred);
    }
    
    Ok(())
}