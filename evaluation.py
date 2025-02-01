import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, test_loader, criterion):
    """
    Avalia o modelo e retorna um dicionário com as métricas
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
                
                # Converter para numpy e flatten
                pred_np = outputs.cpu().numpy().flatten()
                target_np = targets.cpu().numpy().flatten()
                
                predictions.extend(pred_np)
                actuals.extend(target_np)
                
            except Exception as e:
                print(f"Error during inference: {str(e)}")
                continue
    
    # Converter listas para arrays numpy
    predictions = np.array(predictions, dtype=np.float32)
    actuals = np.array(actuals, dtype=np.float32)
    
    # Calcular métricas
    metrics = {
        'overall': {
            'mse': float(mean_squared_error(actuals, predictions)),
            'mae': float(mean_absolute_error(actuals, predictions)),
            'r2': float(r2_score(actuals, predictions)),
            'loss': float(total_loss / len(test_loader))
        },
        'predictions': predictions,
        'actuals': actuals
    }
    
    # Imprimir métricas
    print("\nOverall Metrics:")
    print(f"MSE: {metrics['overall']['mse']:.4f}")
    print(f"MAE: {metrics['overall']['mae']:.4f}")
    print(f"R²: {metrics['overall']['r2']:.4f}")
    
    return metrics 