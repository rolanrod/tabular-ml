// use crate::{Matrix, Vector};
// use ndarray::s;
use crate::{Matrix, Vector};
use ndarray::s;

#[derive(Clone, Debug)]
pub struct Dataset {
    pub features: Matrix,
    pub labels: Vector,
}

impl Dataset {
    pub fn new(features: Matrix, labels: Vector) -> Result<Self, String> {
        if features.nrows() != labels.len() {
            return Err("Numbers of samples in features and targets must match".to_string());
        }

        Ok(Self { features, labels })
    }

    pub fn n_samples(&self) -> usize {
        self.features.nrows()
    }

    pub fn n_features(&self) -> usize {
        self.features.ncols()
    }

    pub fn train_test_split(&self, test_size: f64) -> Result<(Self, Self), String> {
        if test_size <= 0.0 || test_size >= 1.0 {
            return Err("test_size must be between 0 and 1".to_string());
        }

        let n_samples = self.n_samples();
        let n_test = (n_samples as f64 * test_size).round() as usize;
        let n_train = n_samples - n_test;

        let train_features = self.features.slice(s![..n_train, ..]).to_owned();
        let train_labels = self.labels.slice(s![..n_train, ..]).to_owned();

        let test_features = self.features.slice(s![n_test.., ..]).to_owned();
        let test_labels = self.labels.slice(s![n_test.., ..]).to_owned();

        let train_dataset = Dataset::new(train_features, train_labels)?;
        let test_dataset = Dataset::new(test_features, test_labels)?;

        Ok((train_dataset, test_dataset))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_dataset_creation() {
        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let targets = array![1.0, 2.0, 3.0];
        
        let dataset = Dataset::new(features, targets).unwrap();
        assert_eq!(dataset.n_samples(), 3);
        assert_eq!(dataset.n_features(), 2);
    }
    
    #[test]
    fn test_train_test_split() {
        let features = Array2::zeros((100, 5));
        let targets = Vector::zeros(100);
        let dataset = Dataset::new(features, targets).unwrap();
        
        let (train, test) = dataset.train_test_split(0.2).unwrap();
        assert_eq!(train.n_samples(), 80);
        assert_eq!(test.n_samples(), 20);
    }
}