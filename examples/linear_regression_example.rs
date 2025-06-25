use nametbd::{Matrix, Vector, Dataset, LinearRegression, StandardScaler};
use ndarray::Array2;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Load CSV data (example format: feature1,feature2,feature3,target)
    let (features, targets) = load_csv("data/housing.csv")?;
    
    // Step 2: Create dataset
    let dataset = Dataset::new(features, targets)?;
    println!("Dataset: {} samples, {} features", dataset.n_samples(), dataset.n_features());
    
    // Step 3: Split into train/test
    let (train_data, test_data) = dataset.train_test_split(0.2)?;
    
    // Step 4: Preprocessing - standardize features
    let mut scaler = StandardScaler::new();
    let train_features_scaled = scaler.fit_transform(&train_data.features)?;
    let test_features_scaled = scaler.transform(&test_data.features)?;
    
    // Step 5: Train linear regression model
    let mut model = LinearRegression::new();
    model.fit(&train_features_scaled, &train_data.labels)?;
    
    // Step 6: Make predictions
    let train_predictions = model.predict(&train_features_scaled)?;
    let test_predictions = model.predict(&test_features_scaled)?;
    
    // Step 7: Evaluate model
    let train_score = model.score(&train_features_scaled, &train_data.labels)?;
    let test_score = model.score(&test_features_scaled, &test_data.labels)?;
    
    let train_mse = nametbd::metrics::mean_squared_error(&train_data.labels, &train_predictions)?;
    let test_mse = nametbd::metrics::mean_squared_error(&test_data.labels, &test_predictions)?;
    
    println!("Results:");
    println!("  Training R² score: {:.4}", train_score);
    println!("  Test R² score: {:.4}", test_score);
    println!("  Training MSE: {:.4}", train_mse);
    println!("  Test MSE: {:.4}", test_mse);
    
    // Step 8: Inspect model parameters
    if let Some(coeffs) = &model.coefficients {
        println!("  Coefficients: {:?}", coeffs);
        println!("  Intercept: {:.4}", model.intercept.unwrap_or(0.0));
    }
    
    Ok(())
}

fn load_csv(filename: &str) -> Result<(Matrix, Vector), Box<dyn std::error::Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    
    let mut rows: Vec<Vec<f64>> = Vec::new();
    
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 {
            continue; // Skip header
        }
        
        let values: Result<Vec<f64>, _> = line
            .split(',')
            .map(|s| s.trim().parse::<f64>())
            .collect();
        
        rows.push(values?);
    }
    
    if rows.is_empty() {
        return Err("No data found in CSV".into());
    }
    
    let n_samples = rows.len();
    let n_features = rows[0].len() - 1; // Last column is target
    
    let mut features_data = vec![0.0; n_samples * n_features];
    let mut targets_data = vec![0.0; n_samples];
    
    for (i, row) in rows.iter().enumerate() {
        for j in 0..n_features {
            features_data[i * n_features + j] = row[j];
        }
        targets_data[i] = row[n_features]; // Last column is target
    }
    
    let features = Matrix::from_shape_vec((n_samples, n_features), features_data)?;
    let targets = Vector::from(targets_data);
    
    Ok((features, targets))
}