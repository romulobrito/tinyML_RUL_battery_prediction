import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

def evaluate_model(model, test_loader, criterion, scaler_y=None):
    """
    Evaluate the model with specific metrics based on AED
    """
    model.eval()
    predictions = []
    actuals = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            try:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += float(loss.item())
                
                pred_np = outputs.cpu().numpy().flatten()
                target_np = targets.cpu().numpy().flatten()
                
                predictions.extend(pred_np)
                actuals.extend(target_np)
                
            except Exception as e:
                print(f"Error during inference: {str(e)}")
                continue
    
    predictions = np.array(predictions, dtype=np.float32)
    actuals = np.array(actuals, dtype=np.float32)
    
    if scaler_y is not None:
        predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()
    
    # General Metrics
    metrics = {
        'overall': {
            'mse': float(mean_squared_error(actuals, predictions)),
            'mae': float(mean_absolute_error(actuals, predictions)),
            'r2': float(r2_score(actuals, predictions)),
            'loss': float(total_loss / len(test_loader))
        }
    }
    
    # Metrics by RUL bin (based on AED)
    rul_bins = [0, 10, 100, 500, 1000, float('inf')]
    rul_labels = ['0-10', '11-100', '101-500', '501-1000', '>1000']
    
    df_results = pd.DataFrame({
        'actual': actuals,
        'predicted': predictions
    })
    df_results['rul_bin'] = pd.cut(df_results['actual'], bins=rul_bins, labels=rul_labels)
    
    metrics['by_rul_range'] = {}
    for rul_range in rul_labels:
        mask = df_results['rul_bin'] == rul_range
        if mask.any():
            range_actuals = df_results.loc[mask, 'actual']
            range_preds = df_results.loc[mask, 'predicted']
            metrics['by_rul_range'][rul_range] = {
                'mse': float(mean_squared_error(range_actuals, range_preds)),
                'mae': float(mean_absolute_error(range_actuals, range_preds)),
                'samples': int(mask.sum()),
                'mean_error': float((range_preds - range_actuals).mean())
            }
    
    # Extreme Error Analysis (based on identified outliers)
    error = predictions - actuals
    q1 = np.percentile(error, 25)
    q3 = np.percentile(error, 75)
    iqr = q3 - q1
    outlier_mask = (error < (q1 - 1.5 * iqr)) | (error > (q3 + 1.5 * iqr))
    
    metrics['error_analysis'] = {
        'mean_error': float(error.mean()),
        'std_error': float(error.std()),
        'outlier_predictions_pct': float((outlier_mask.sum() / len(error)) * 100),
        'max_overestimation': float(error.max()),
        'max_underestimation': float(error.min())
    }
    
    print("\n=== General Metrics ===")
    print(f"MSE: {metrics['overall']['mse']:.4f}")
    print(f"MAE: {metrics['overall']['mae']:.4f}")
    print(f"R²: {metrics['overall']['r2']:.4f}")
    
    print("\n=== Métricas por Faixa de RUL ===")
    for rul_range, range_metrics in metrics['by_rul_range'].items():
        print(f"\nRange {rul_range}:")
        print(f"Samples: {range_metrics['samples']}")
        print(f"MAE: {range_metrics['mae']:.4f}")
        print(f"Mean Error: {range_metrics['mean_error']:.4f}")
    
    print("\n=== Error Analysis ===")
    print(f"Mean Error: {metrics['error_analysis']['mean_error']:.4f}")
    print(f"Standard Deviation of Error: {metrics['error_analysis']['std_error']:.4f}")
    print(f"% of Outlier Predictions: {metrics['error_analysis']['outlier_predictions_pct']:.2f}%")
    
    return metrics
