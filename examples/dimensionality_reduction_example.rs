use nametbd::{PCA, LDA, TruncatedSVD, StandardScaler, Matrix, Vector};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Dimensionality Reduction Techniques Comparison ===\n");
    
    // Create sample high-dimensional data
    let x = array![
        [1.0, 2.0, 3.0, 4.0, 5.0, 1.1, 2.1, 3.1],
        [2.0, 4.0, 6.0, 8.0, 10.0, 2.2, 4.2, 6.2],
        [3.0, 6.0, 9.0, 12.0, 15.0, 3.3, 6.3, 9.3],
        [1.5, 3.0, 4.5, 6.0, 7.5, 1.6, 3.1, 4.6],
        [10.0, 8.0, 6.0, 4.0, 2.0, 10.5, 8.5, 6.5],
        [12.0, 10.0, 8.0, 6.0, 4.0, 12.5, 10.5, 8.5],
        [14.0, 12.0, 10.0, 8.0, 6.0, 14.5, 12.5, 10.5],
        [11.0, 9.0, 7.0, 5.0, 3.0, 11.5, 9.5, 7.5],
        [20.0, 15.0, 10.0, 5.0, 0.0, 20.5, 15.5, 10.5],
        [25.0, 20.0, 15.0, 10.0, 5.0, 25.5, 20.5, 15.5]
    ];
    
    // Create corresponding labels for supervised methods
    let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0];
    
    println!("Original data shape: {} samples, {} features", x.nrows(), x.ncols());
    let unique_classes: Vec<i32> = y.iter().map(|&x| x as i32).collect::<std::collections::HashSet<_>>().into_iter().collect();
    println!("Classes: {:?}\n", unique_classes);
    
    // Standardize the data
    let mut scaler = StandardScaler::new();
    let x_scaled = scaler.fit_transform(&x)?;
    
    println!("=== Principal Component Analysis (PCA) ===");
    println!("PCA is an unsupervised technique that finds directions of maximum variance");
    
    // Test PCA with different numbers of components
    for &n_components in &[2, 3, 4] {
        match test_pca(&x_scaled, n_components) {
            Ok(msg) => println!("{}", msg),
            Err(e) => println!("PCA with {} components failed: {}", n_components, e),
        }
    }
    
    println!("\n=== Linear Discriminant Analysis (LDA) ===");
    println!("LDA is a supervised technique that finds directions that best separate classes");
    
    // Test LDA
    match test_lda(&x_scaled, &y) {
        Ok(msg) => println!("{}", msg),
        Err(e) => println!("LDA failed: {}", e),
    }
    
    println!("\n=== Truncated SVD ===");
    println!("Truncated SVD decomposes the data matrix using singular value decomposition");
    
    // Test Truncated SVD with different numbers of components
    for &n_components in &[2, 3, 4] {
        match test_truncated_svd(&x_scaled, n_components) {
            Ok(msg) => println!("{}", msg),
            Err(e) => println!("TruncatedSVD with {} components failed: {}", n_components, e),
        }
    }
    
    println!("\n=== Comparative Analysis ===");
    
    // Compare reconstruction quality for different methods
    println!("Comparing reconstruction quality (lower MSE is better):");
    println!("{:<20} {:>12} {:>15}", "Method", "Components", "Reconstruction MSE");
    println!("{}", "-".repeat(50));
    
    // PCA reconstruction
    for &n_comp in &[2, 3] {
        if let Ok(mse) = compute_pca_reconstruction_error(&x_scaled, n_comp) {
            println!("{:<20} {:>12} {:>15.6}", "PCA", n_comp, mse);
        }
    }
    
    // SVD reconstruction  
    for &n_comp in &[2, 3] {
        if let Ok(mse) = compute_svd_reconstruction_error(&x_scaled, n_comp) {
            println!("{:<20} {:>12} {:>15.6}", "Truncated SVD", n_comp, mse);
        }
    }
    
    println!("\n=== Practical Use Cases ===");
    println!("• PCA: Exploratory data analysis, noise reduction, feature extraction");
    println!("• LDA: Classification preprocessing, supervised dimensionality reduction");
    println!("• Truncated SVD: Large sparse matrices, recommender systems, text analysis");
    println!("• All methods: Visualization (reduce to 2D/3D), compression, denoising");
    
    println!("\n=== Implementation Notes ===");
    println!("These implementations use simplified eigenvalue decomposition algorithms.");
    println!("For production use, consider libraries with optimized LAPACK/BLAS backends.");
    println!("The algorithms demonstrate the core concepts and API patterns.");
    
    Ok(())
}

fn test_pca(x: &Matrix, n_components: usize) -> Result<String, String> {
    let mut pca = PCA::new().n_components(n_components);
    
    let transformed = pca.fit_transform(x)?;
    
    let _components_shape = pca.components.as_ref().map(|c| c.shape()).unwrap_or(&[0, 0]);
    let explained_var = pca.explained_variance_ratio.as_ref()
        .map(|ev| ev.sum())
        .unwrap_or(0.0);
    
    Ok(format!(
        "PCA({} components): Output shape {:?}, Explained variance: {:.4}", 
        n_components, transformed.shape(), explained_var
    ))
}

fn test_lda(x: &Matrix, y: &Vector) -> Result<String, String> {
    let mut lda = LDA::new();
    
    let transformed = lda.fit_transform(x, y)?;
    
    let _components_shape = lda.components.as_ref().map(|c| c.shape()).unwrap_or(&[0, 0]);
    let explained_var = lda.explained_variance_ratio.as_ref()
        .map(|ev| ev.sum())
        .unwrap_or(0.0);
    
    // Test classification accuracy
    let accuracy = lda.score(x, y).unwrap_or(0.0);
    
    Ok(format!(
        "LDA: Output shape {:?}, Explained variance: {:.4}, Classification accuracy: {:.4}", 
        transformed.shape(), explained_var, accuracy
    ))
}

fn test_truncated_svd(x: &Matrix, n_components: usize) -> Result<String, String> {
    let mut svd = TruncatedSVD::new(n_components);
    
    let transformed = svd.fit_transform(x)?;
    
    let explained_var = svd.explained_variance_ratio.as_ref()
        .map(|ev| ev.sum())
        .unwrap_or(0.0);
    
    Ok(format!(
        "TruncatedSVD({} components): Output shape {:?}, Explained variance: {:.4}", 
        n_components, transformed.shape(), explained_var
    ))
}

fn compute_pca_reconstruction_error(x: &Matrix, n_components: usize) -> Result<f64, String> {
    let mut pca = PCA::new().n_components(n_components);
    let transformed = pca.fit_transform(x)?;
    let reconstructed = pca.inverse_transform(&transformed)?;
    
    let diff = x - &reconstructed;
    let mse = diff.mapv(|x| x * x).mean().unwrap_or(f64::INFINITY);
    Ok(mse)
}

fn compute_svd_reconstruction_error(x: &Matrix, n_components: usize) -> Result<f64, String> {
    let mut svd = TruncatedSVD::new(n_components);
    let transformed = svd.fit_transform(x)?;
    let reconstructed = svd.inverse_transform(&transformed)?;
    
    let diff = x - &reconstructed;
    let mse = diff.mapv(|x| x * x).mean().unwrap_or(f64::INFINITY);
    Ok(mse)
}