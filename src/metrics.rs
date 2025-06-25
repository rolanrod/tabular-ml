use crate::{Vector, Matrix};

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

pub fn accuracy_score(y_true: &Vector, y_pred: &Vector) -> Result<f64, String> {
    if y_true.len() != y_pred.len() {
        return Err("y_true and y_pred must have the same length".to_string());
    }
    
    let correct = y_true.iter()
        .zip(y_pred.iter())
        .map(|(&true_val, &pred_val)| if (true_val - pred_val).abs() < 1e-10 { 1.0 } else { 0.0 })
        .sum::<f64>();
    
    Ok(correct / y_true.len() as f64)
}

pub fn precision_score(y_true: &Vector, y_pred: &Vector) -> Result<f64, String> {
    if y_true.len() != y_pred.len() {
        return Err("y_true and y_pred must have the same length".to_string());
    }
    
    let (tp, fp) = calculate_tp_fp(y_true, y_pred);
    
    if tp + fp == 0.0 {
        return Ok(0.0);
    }
    
    Ok(tp / (tp + fp))
}

pub fn recall_score(y_true: &Vector, y_pred: &Vector) -> Result<f64, String> {
    if y_true.len() != y_pred.len() {
        return Err("y_true and y_pred must have the same length".to_string());
    }
    
    let (tp, fn_val) = calculate_tp_fn(y_true, y_pred);
    
    if tp + fn_val == 0.0 {
        return Ok(0.0);
    }
    
    Ok(tp / (tp + fn_val))
}

pub fn f1_score(y_true: &Vector, y_pred: &Vector) -> Result<f64, String> {
    let precision = precision_score(y_true, y_pred)?;
    let recall = recall_score(y_true, y_pred)?;
    
    if precision + recall == 0.0 {
        return Ok(0.0);
    }
    
    Ok(2.0 * precision * recall / (precision + recall))
}

pub fn confusion_matrix(y_true: &Vector, y_pred: &Vector) -> Result<Matrix, String> {
    if y_true.len() != y_pred.len() {
        return Err("y_true and y_pred must have the same length".to_string());
    }
    
    let mut tn = 0.0;
    let mut fp = 0.0;
    let mut fn_val = 0.0;
    let mut tp = 0.0;
    
    for (&true_val, &pred_val) in y_true.iter().zip(y_pred.iter()) {
        let true_binary = if true_val > 0.5 { 1.0 } else { 0.0 };
        let pred_binary = if pred_val > 0.5 { 1.0 } else { 0.0 };
        
        match (true_binary, pred_binary) {
            (0.0, 0.0) => tn += 1.0,
            (0.0, 1.0) => fp += 1.0,
            (1.0, 0.0) => fn_val += 1.0,
            (1.0, 1.0) => tp += 1.0,
            _ => {}
        }
    }
    
    let cm = Matrix::from_shape_vec((2, 2), vec![tn, fp, fn_val, tp])
        .map_err(|_| "Failed to create confusion matrix".to_string())?;
    
    Ok(cm)
}

fn calculate_tp_fp(y_true: &Vector, y_pred: &Vector) -> (f64, f64) {
    let mut tp = 0.0;
    let mut fp = 0.0;
    
    for (&true_val, &pred_val) in y_true.iter().zip(y_pred.iter()) {
        let true_binary = if true_val > 0.5 { 1.0 } else { 0.0 };
        let pred_binary = if pred_val > 0.5 { 1.0 } else { 0.0 };
        
        if pred_binary == 1.0 {
            if true_binary == 1.0 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
        }
    }
    
    (tp, fp)
}

fn calculate_tp_fn(y_true: &Vector, y_pred: &Vector) -> (f64, f64) {
    let mut tp = 0.0;
    let mut fn_val = 0.0;
    
    for (&true_val, &pred_val) in y_true.iter().zip(y_pred.iter()) {
        let true_binary = if true_val > 0.5 { 1.0 } else { 0.0 };
        let pred_binary = if pred_val > 0.5 { 1.0 } else { 0.0 };
        
        if true_binary == 1.0 {
            if pred_binary == 1.0 {
                tp += 1.0;
            } else {
                fn_val += 1.0;
            }
        }
    }
    
    (tp, fn_val)
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

    #[test]
    fn test_accuracy_score() {
        let y_true = array![1.0, 0.0, 1.0, 1.0];
        let y_pred = array![1.0, 0.0, 1.0, 0.0];
        
        let accuracy = accuracy_score(&y_true, &y_pred).unwrap();
        assert!((accuracy - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_precision_score() {
        let y_true = array![1.0, 0.0, 1.0, 1.0];
        let y_pred = array![1.0, 0.0, 1.0, 0.0];
        
        let precision = precision_score(&y_true, &y_pred).unwrap();
        assert!((precision - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_recall_score() {
        let y_true = array![1.0, 0.0, 1.0, 1.0];
        let y_pred = array![1.0, 0.0, 1.0, 0.0];
        
        let recall = recall_score(&y_true, &y_pred).unwrap();
        assert!((recall - 2.0/3.0).abs() < 1e-10);
    }

    #[test]
    fn test_f1_score() {
        let y_true = array![1.0, 0.0, 1.0, 1.0];
        let y_pred = array![1.0, 0.0, 1.0, 0.0];
        
        let f1 = f1_score(&y_true, &y_pred).unwrap();
        let expected = 2.0 * 1.0 * (2.0/3.0) / (1.0 + 2.0/3.0);
        assert!((f1 - expected).abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix() {
        let y_true = array![1.0, 0.0, 1.0, 0.0];
        let y_pred = array![1.0, 0.0, 0.0, 0.0];
        
        let cm = confusion_matrix(&y_true, &y_pred).unwrap();
        assert_eq!(cm.shape(), &[2, 2]);
        assert_eq!(cm[(0, 0)], 2.0); // TN
        assert_eq!(cm[(0, 1)], 0.0); // FP
        assert_eq!(cm[(1, 0)], 1.0); // FN
        assert_eq!(cm[(1, 1)], 1.0); // TP
    }
}