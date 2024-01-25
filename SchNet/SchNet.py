import sys
sys.path.append('../Utilities') 
import utils
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from torch_geometric.nn.models import SchNet
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def fit(model, train_loader, val_loader, epochs):
    model.to(device)  # Mueve el modelo a la GPU si está disponible
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_values = []

    mse_metric = MeanSquaredError()
    mae_metric = MeanAbsoluteError()
    r2score = R2Score()

    mean_train_mse = None
    mean_train_mae = None
    mean_train_r2 = None
    mean_val_mse = None
    mean_val_mae = None
    mean_val_r2 = None

    model.train()
    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)

    for epoch in range(epochs):
        total_train_r2 = 0.0
        total_train_mse = 0.0
        total_train_mae = 0.0

        for train_data in train_loader:
            train_data = train_data.to(device)
            train_out = model(train_data.z, train_data.pos, train_data.batch)
            train_loss = criterion(train_out.view(-1), train_data.y)

            train_r2 = r2score(train_out.view(-1), train_data.y)
            train_mse = mse_metric(train_out.view(-1), train_data.y)
            train_mae = mae_metric(train_out.view(-1), train_data.y)

            total_train_r2 += train_r2.item()
            total_train_mse += train_mse.item()
            total_train_mae += train_mae.item()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        #if epoch % 50 == 0:
        loss_values.append(float(train_loss.item()))
        
        total_val_r2 = 0.0
        total_val_mse = 0.0
        total_val_mae = 0.0            
        
        with torch.no_grad():
            for val_data in val_loader:
                val_data = val_data.to(device)
                val_out = model(val_data.z, val_data.pos, val_data.batch)    
                val_loss = criterion(val_out.view(-1), val_data.y)
                val_r2 = r2score(val_out.view(-1), val_data.y)
                val_mse = mse_metric(val_out.view(-1), val_data.y)
                val_mae = mae_metric(val_out.view(-1), val_data.y)

                total_val_r2 += val_r2.item()
                total_val_mse += val_mse.item()
                total_val_mae += val_mae.item()

        mean_train_mse = total_train_mse / num_train_batches
        mean_train_mae = total_train_mae / num_train_batches
        mean_train_r2 = total_train_r2 / num_train_batches

        mean_val_mse = total_val_mse / num_val_batches
        mean_val_mae = total_val_mae / num_val_batches
        mean_val_r2 = total_val_r2 / num_val_batches

        print(f"Epoch [{epoch+1}/{epochs}], "
                f"Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}, "
                f"Train MSE: {mean_train_mse:.4f}, Val MSE: {mean_val_mse:.4f}, "
                f"Train MAE: {mean_train_mae:.4f}, Val MAE: {mean_val_mae:.4f}, "
                f"Train R2: {mean_train_r2:.4f}, Val R2: {mean_val_r2:.4f}")

    return mean_train_mse, mean_train_mae, mean_train_r2, mean_val_mse, mean_val_mae, mean_val_r2, loss_values
def test(model, test_loader):
    criterion = nn.MSELoss()
    total_loss = 0
    mse_metric = MeanSquaredError()
    mae_metric = MeanAbsoluteError()
    r2score = R2Score()
    
    total_r2 = 0.0
    total_mse = 0.0
    total_mae = 0.0         
    model.eval()
    y_pred = []
    y_test = []        
    num_batches = len(test_loader)
    for data in test_loader:
        data = data.to(device)
        test_output = model(data.z, data.pos,data.batch)
        y_test.extend(data.y.cpu().tolist())
        y_pred.extend(test_output.view(-1).tolist())            
        loss = criterion(test_output.view(-1),data.y)
        r2 = r2score(test_output.view(-1), data.y)
        mse = mse_metric(test_output.view(-1), data.y)
        mae = mae_metric(test_output.view(-1), data.y)                
        total_r2 += r2.item()
        total_mse += mse.item()
        total_mae += mae.item()  
        total_loss += loss.item() 
    average_loss = total_loss / num_batches
    mean_mse = total_mae / num_batches
    mean_mae = total_mse / num_batches
    mean_r2 = total_r2 / num_batches
    return y_pred,y_test,mean_mse, mean_mae, mean_r2,average_loss      
#Preparar el dataset
def prepare_data(data, target=5):
    return Data(z=data.z, pos=data.pos, y=data.y[:, target].view(-1))
def main(args):
    k_sizes = [3000, 5000]
    # k_sizes = [3000, 5000, 10000, 30000, 50000, 10000, 130000]
    all_loss_values = []
    stats_all = []
    dataset = QM9(root="../../Dataset/QM9")
    dataset.transform = prepare_data
    print(dataset.y[1,5])
    for k in k_sizes:
        train_dataset, val_dataset, test_dataset = utils.split_dataset(dataset, k)
        train_loader, val_loader, test_loader = utils.create_data_loaders(train_dataset, val_dataset, test_dataset, args.batch_size)
        schnet = SchNet()
        mean_train_mse, mean_train_mae, mean_train_r2, mean_val_mse, mean_val_mae, mean_val_r2, loss_values = fit(schnet,train_loader, val_loader=val_loader, epochs=args.epochs)

        all_loss_values.append(loss_values)
        print(f'K = {k}')
        print(f'Train Loss')
        print(f'Mean MSE: {mean_train_mse:.4f}')
        print(f'Mean MAE: {mean_train_mae:.4f}')
        print(f'Mean R²: {mean_train_r2:.4f}')
        
        print(f'Validation Loss')
        print(f'Mean MSE: {mean_val_mse:.4f}')
        print(f'Mean MAE: {mean_val_mae:.4f}')
        print(f'Mean R²: {mean_val_r2:.4f}')
        
        y_pred, y_test, mean_mse_t, mean_mae_t, mean_r2_t,average_loss = test(schnet,test_loader)
        print(f"Test Loss:")
        print(f'Mean MSE: {mean_mse_t:.4f}')
        print(f'Mean MAE: {mean_mae_t:.4f}')
        print(f'Mean R²: {mean_r2_t:.4f}')

        stats = {'k': k, 'Mean MSE': mean_train_mse, 'Mean MAE': mean_train_mae, 'Mean R²': mean_train_r2,
                       'Validation Mean MSE': mean_val_mse, 'Validation Mean MAE': mean_val_mae,
                       'Validation Mean R²': mean_val_r2,
                       'Test Mean MSE': mean_mse_t, 'Test Mean MAE': mean_mae_t, 'Test Mean R²': mean_r2_t}
        stats_all.append(stats)
        utils.plot_scatter(y_test, y_pred, k=k,arq_name = 'SchNet')

        residuals = [actual - predicted for actual, predicted in zip(y_test, y_pred)]
        utils.plot_residuals(y_pred, residuals, k=k,arq_name = 'SchNet')

    # Gráfica de pérdida durante el entrenamiento
    utils.guardar_lista_diccionarios_a_csv(stats_all, 'SchNet')
    utils.plot_loss_convergence(all_loss_values, list(map(str, k_sizes)), 'SchNet')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenar una red SchNet')
    parser.add_argument('--n', type=int, default=30000, help='Tamaño del conjunto de datos')
    parser.add_argument('--data_train_size', type=float, default=0.7, help='Proporción de datos de entrenamiento')
    parser.add_argument('--data_val_size', type=float, default=0.15, help='Proporción de datos de validacion')
    parser.add_argument('--data_test_size', type=float, default=0.30, help='Proporción de datos de prueba')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamaño del lote (batch_size)')
    parser.add_argument('--hidden_size', type=int, default=128, help='Dimensión oculta (hidden_size)')    
    parser.add_argument('--target_index', type=int, default= 5, help='target')
    parser.add_argument('--epochs', type=int, default= 12, help='Numero de epocas')
    args = parser.parse_args()
    main(args)
