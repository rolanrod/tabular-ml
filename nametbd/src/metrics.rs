use crate::Vector;

pub fn mean_squared_error(y_true: &Vector, y_pred: &Vector) -> Result<f64, String> {
    if y_true.len() != y_pred.len() {
        return Err("y_true and y_pred must have the same length".to_string());
    }
    
    let diff = y_true - y_pred;
    let mse = diff.mapv(|x| x * x).mean().unwrap();
    Ok(mse)
}

pub fn mean_absolute_error(y_true: &Vector, y_pred: &Vector) -> Result<f64, String> {
    if y_true.len() != y_pred.len() {
        return Err("y_true and y_pred must have the same length".to_string());
    }
    
    let diff = y_true - y_pred;
    let mae = diff.mapv(|x| x.abs()).mean().unwrap();
    Ok(mae)
}

pub fn r2_score(y_true: &Vector, y_pred: &Vector) -> Result<f64, String> {
    if y_true.len() != y_pred.len() {
        return Err("y_true and y_pred must have the same length".to_string());
    }
    
    let y_mean = y_true.mean().unwrap();
    let ss_res = (y_true - y_pred).mapv(|x| x * x).sum();
    let ss_tot = y_true.mapv(|x| (x - y_mean) * (x - y_mean)).sum();
    
    if ss_tot == 0.0 {
        return Ok(1.0); // Perfect prediction when variance is zero
    }
    
    Ok(1.0 - ss_res / ss_tot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_mean_squared_error() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        
        let mse = mean_squared_error(&y_true, &y_pred).unwrap();
        assert!((mse - 0.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_r2_score() {
        let y_true = array![1.0, 2.0, 3.0, 4.0];
        let y_pred = array![1.0, 2.0, 3.0, 4.0];
        
        let r2 = r2_score(&y_true, &y_pred).unwrap();
        assert!((r2 - 1.0).abs() < 1e-10);
    }
}