from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import torch
import matplotlib.pyplot as plt

def predict_and_plot(model, data_loader, title, scaler, device):
    model = model.to(device)
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)

            predictions.append(outputs.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0).reshape(-1, 14)
    actuals = np.concatenate(actuals, axis=0).reshape(-1, 14)
    print(predictions.shape)
    print(actuals.shape)

    predictions_rescaled = scaler.inverse_transform(predictions)
    actuals_rescaled = scaler.inverse_transform(actuals)
    mse = mean_squared_error(actuals_rescaled[:, -1], predictions_rescaled[:, -1])
    mae = mean_absolute_error(actuals_rescaled[:, -1], predictions_rescaled[:, -1])
    print(actuals_rescaled[:, -1].shape)

    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    # print(len(predictions_rescaled))
    # print(len(actuals_rescaled))

    plt.figure(figsize=(10, 6))
    plt.plot(actuals_rescaled[10], label='True', color='blue')
    plt.plot(predictions_rescaled[10], label='Predicted', color='orange')
    plt.title(title)
    plt.legend()
    plt.savefig("/public/home/djingwang/pyyu/CapitalBikeshare/prediction.jpg")
    plt.show()

