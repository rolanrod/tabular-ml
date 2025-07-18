pub use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

pub mod dataset;
pub mod preprocessing;
pub mod linear_model;
pub mod metrics;
pub mod decomposition;
pub mod cluster;

pub use dataset::Dataset;
pub use preprocessing::StandardScaler;
pub use linear_model::{LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, SVC, Kernel};
pub use decomposition::{PCA, LDA, TruncatedSVD};
pub use cluster::{KMeans, DBSCAN};

pub type Vector = Array1<f64>;
pub type Matrix = Array2<f64>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_types_work() {
        let vec = Vector::zeros(5);
        let mat = Matrix::zeros((3, 4));
        assert_eq!(vec.len(), 5);
        assert_eq!(mat.shape(), &[3, 4]);
    }
}
