from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import torch
import matplotlib.pyplot as plt

def mse_mae(model, data_loader, scaler, device, model_name, output_window, exp_id):
    model = model.to(device)
    model.eval()
    predictions, actuals = [], []
    mse_list, mae_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            mse_per_step, mae_per_step = [], []

            X_batch = X_batch.to(device)    # torch.Size([32, 96, 1])
            outputs = model(X_batch)
            output_rescaled = scaler.inverse_transform(outputs.cpu().numpy())
            # print(output_rescaled[0])
            predictions.append(output_rescaled)
            target_rescaled = scaler.inverse_transform(y_batch.cpu().numpy())
            # print(target_rescaled[0])
            # exit()
            actuals.append(target_rescaled)
            mse = mean_squared_error(target_rescaled, output_rescaled)
            mae = mean_absolute_error(target_rescaled, output_rescaled)
            mse_per_step.append(mse)
            mae_per_step.append(mae)

            mse_list.append(mse_per_step)
            mae_list.append(mae_per_step)
    # print(f'Mean Squared Error (MSE): {np.average(mse_list, axis=0)}')
    print(f'Mean Squared Error (MSE): {np.average(mse_list)}')
    # print(f'Mean Absolute Error (MAE): {np.average(mae_list, axis=0)}')
    print(f'Mean Absolute Error (MAE): {np.average(mae_list)}')


    plt.figure(figsize=(10, 6))
    plt.plot(actuals[0][0], label='Target', color='blue')
    plt.plot(predictions[0][0], label='Prediction', color='orange')
    plt.title('Bicycle Rental Count (cnt)')
    plt.legend()
    plt.savefig(f"/public/home/djingwang/pyyu/CapitalBikeshare/picture/{model_name}_{output_window}h_{exp_id}.svg")
    plt.show()

def mse_mae_RevDimTSFormer(model, data_loader, scaler, device, model_name, output_window, exp_id):
    model = model.to(device)
    model.eval()
    predictions, actuals = [], []
    mse_list, mae_list = [], []
    with torch.no_grad():
        for X_batch, X_mark_batch, X_dec_batch, X_dec_mark_batch, y_batch in data_loader:
            mse_per_step, mae_per_step = [], []

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device) 
            X_mark_batch = X_mark_batch.to(device)
            X_dec_batch = X_dec_batch.to(device) 
            X_dec_mark_batch = X_dec_mark_batch.to(device)

            outputs = model(X_batch, X_mark_batch, X_dec_batch, X_dec_mark_batch)
            output_rescaled = scaler.inverse_transform(outputs.cpu().numpy())
            predictions.append(output_rescaled)
            target_rescaled = scaler.inverse_transform(y_batch.cpu().numpy())
            actuals.append(target_rescaled)
            mse = mean_squared_error(target_rescaled, output_rescaled)
            mae = mean_absolute_error(target_rescaled, output_rescaled)
            mse_per_step.append(mse)
            mae_per_step.append(mae)

            mse_list.append(mse_per_step)
            mae_list.append(mae_per_step)
    # print(f'Mean Squared Error (MSE): {np.average(mse_list, axis=0)}')
    print(f'Mean Squared Error (MSE): {np.average(mse_list)}')
    # print(f'Mean Absolute Error (MAE): {np.average(mae_list, axis=0)}')
    print(f'Mean Absolute Error (MAE): {np.average(mae_list)}')


    plt.figure(figsize=(10, 6))
    plt.plot(actuals[0][0], label='Target', color='blue')
    plt.plot(predictions[0][0], label='Prediction', color='orange')
    plt.title('Bicycle Rental Count (cnt)')
    plt.legend()
    plt.savefig(f"/public/home/djingwang/pyyu/CapitalBikeshare/picture/{model_name}_{output_window}h_{exp_id}.svg")
    plt.show()

    return np.average(mse_list)
