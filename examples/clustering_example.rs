use nametbd::{KMeans, DBSCAN, StandardScaler, Matrix, Vector};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Clustering Algorithms Comparison ===\n");
    
    // Create sample data with three natural clusters
    let x = array![
        // Cluster 1: around (2, 2)
        [1.5, 1.8], [2.0, 2.2], [2.3, 1.9], [1.8, 2.5], [2.1, 1.7],
        // Cluster 2: around (8, 8)  
        [7.8, 8.2], [8.1, 7.9], [8.3, 8.1], [7.9, 8.4], [8.2, 7.7],
        // Cluster 3: around (2, 8)
        [1.9, 7.8], [2.2, 8.1], [1.7, 8.3], [2.4, 7.9], [2.0, 8.2],
        // Some noise points
        [5.0, 5.0], [0.0, 0.0], [10.0, 0.0]
    ];
    
    println!("Dataset: {} samples, {} features", x.nrows(), x.ncols());
    println!("Expected: 3 natural clusters + some noise points\n");
    
    // Standardize the data
    let mut scaler = StandardScaler::new();
    let x_scaled = scaler.fit_transform(&x)?;
    
    println!("=== K-Means Clustering ===");
    println!("K-Means partitions data into k clusters using centroids");
    
    // Test K-Means with different numbers of clusters
    for &k in &[2, 3, 4, 5] {
        match test_kmeans(&x_scaled, k) {
            Ok(result) => println!("{}", result),
            Err(e) => println!("K-Means(k={}) failed: {}", k, e),
        }
    }
    
    println!("\n=== DBSCAN Clustering ===");
    println!("DBSCAN finds clusters of varying shapes and identifies outliers");
    
    // Test DBSCAN with different parameters
    let dbscan_configs = vec![
        (0.3, 2, "Tight clusters"),
        (0.5, 2, "Medium density"),
        (1.0, 2, "Loose clusters"),
        (0.5, 3, "Higher min_samples"),
    ];
    
    for &(eps, min_samples, description) in &dbscan_configs {
        match test_dbscan(&x_scaled, eps, min_samples) {
            Ok(result) => println!("DBSCAN(eps={}, min_samples={}): {} - {}", 
                                 eps, min_samples, description, result),
            Err(e) => println!("DBSCAN(eps={}, min_samples={}) failed: {}", 
                              eps, min_samples, e),
        }
    }
    
    println!("\n=== Detailed Analysis ===");
    
    // Best K-Means result
    let mut best_kmeans = KMeans::new(3);
    best_kmeans.fit(&x_scaled)?;
    let kmeans_labels = best_kmeans.labels.as_ref().unwrap();
    
    println!("K-Means (k=3) results:");
    println!("  Inertia (within-cluster sum of squares): {:.4}", 
             best_kmeans.inertia.unwrap());
    print_cluster_summary("K-Means", kmeans_labels);
    
    // Best DBSCAN result
    let mut best_dbscan = DBSCAN::new(0.5, 2);
    best_dbscan.fit(&x_scaled)?;
    let dbscan_labels = best_dbscan.labels.as_ref().unwrap();
    
    println!("\nDBSCAN (eps=0.5, min_samples=2) results:");
    println!("  Number of clusters found: {}", 
             best_dbscan.get_n_clusters().unwrap());
    println!("  Number of noise points: {}", 
             best_dbscan.get_n_noise_points().unwrap());
    println!("  Number of core samples: {}", 
             best_dbscan.core_sample_indices.as_ref().unwrap().len());
    print_cluster_summary("DBSCAN", dbscan_labels);
    
    println!("\n=== Algorithm Comparison ===");
    println!("K-Means:");
    println!("  ✓ Fast and efficient");
    println!("  ✓ Good for spherical clusters");
    println!("  ✗ Requires pre-specified k");
    println!("  ✗ Sensitive to outliers");
    println!("  ✗ Assumes similar cluster sizes");
    
    println!("\nDBSCAN:");
    println!("  ✓ Automatically determines number of clusters");
    println!("  ✓ Robust to outliers (identifies noise)");
    println!("  ✓ Can find arbitrary shaped clusters");
    println!("  ✗ Sensitive to hyperparameters (eps, min_samples)");
    println!("  ✗ Struggles with varying densities");
    
    println!("\n=== Use Cases ===");
    println!("• K-Means: Customer segmentation, image compression, market research");
    println!("• DBSCAN: Anomaly detection, image processing, spatial data analysis");
    
    Ok(())
}

fn test_kmeans(x: &Matrix, k: usize) -> Result<String, String> {
    let mut kmeans = KMeans::new(k).max_iter(100);
    let labels = kmeans.fit_predict(x)?;
    
    let inertia = kmeans.inertia.unwrap();
    let unique_labels = count_unique_clusters(&labels);
    
    Ok(format!(
        "K-Means(k={}): {} clusters, Inertia: {:.4}",
        k, unique_labels, inertia
    ))
}

fn test_dbscan(x: &Matrix, eps: f64, min_samples: usize) -> Result<String, String> {
    let mut dbscan = DBSCAN::new(eps, min_samples);
    let labels = dbscan.fit_predict(x)?;
    
    let n_clusters = dbscan.get_n_clusters().unwrap();
    let n_noise = dbscan.get_n_noise_points().unwrap();
    
    Ok(format!(
        "{} clusters, {} noise points",
        n_clusters, n_noise
    ))
}

fn count_unique_clusters(labels: &Vector) -> usize {
    let unique_labels: std::collections::HashSet<i32> = labels.iter()
        .map(|&x| x as i32)
        .filter(|&x| x >= 0) // Exclude noise points (-1)
        .collect();
    unique_labels.len()
}

fn print_cluster_summary(algorithm: &str, labels: &Vector) {
    let unique_labels: std::collections::HashSet<i32> = labels.iter()
        .map(|&x| x as i32)
        .collect();
        
    println!("  {} cluster assignments:", algorithm);
    for &cluster_id in &unique_labels {
        let count = labels.iter()
            .filter(|&&x| x as i32 == cluster_id)
            .count();
        
        if cluster_id == -1 {
            println!("    Noise: {} points", count);
        } else {
            println!("    Cluster {}: {} points", cluster_id, count);
        }
    }
}