use crate::{Matrix, Vector};


pub struct StandardScaler {
    mean: Option<Vector>,
    std: Option<Vector>,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            mean: None,
            std: None,
        }
    }

    pub fn fit(&mut self, data: &Matrix) -> Result<(), String> {
        let mean = data.mean_axis(ndarray::Axis(0))
            .ok_or("Failed to compute mean")?;
        let std = data.std_axis(ndarray::Axis(0), 0.0);

        self.mean = Some(mean);
        self.std = Some(std);
        Ok(())
    }

    pub fn transform(&self, data: &Matrix) -> Result<Matrix, String> {
        let mean = self.mean.as_ref()
            .ok_or("Scaler not fitted. Call fit() first.")?;
        let std = self.std.as_ref()
            .ok_or("Scaler not fitted. Call fit() first.")?;

        let mut result = data.clone();
        for mut row in result.axis_iter_mut(ndarray::Axis(0)) {
            row -= mean;
            row /= std;
        }

        Ok(result)
    }

    pub fn fit_transform(&mut self, data: &Matrix) -> Result<Matrix, String> {
        self.fit(data)?;
        self.transform(data)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_standard_scaler() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut scaler = StandardScaler::new();
        
        let scaled = scaler.fit_transform(&data).unwrap();
        assert_eq!(scaled.shape(), data.shape());
    }
}