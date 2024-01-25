# !/usr/bin/python3

import sys
sys.path.append('../Utilities') 
import utils

import argparse
import torch
import torch_geometric as pyg
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import QM9
from torch_geometric.nn import GATv2Conv
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GAT(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size, heads=16, target_index=5):        
        super().__init__()
        self.target_index = target_index
        self.gat1 = GATv2Conv(in_size, hidden_size, heads=heads)
        self.gat2 = GATv2Conv(hidden_size * heads, hidden_size, heads=1)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )
        # Inicializar tensores con xavier_uniform
        self.init_weights()

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)


    def forward(self, x, edge_index, batch):
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        h = self.gat1(x, edge_index)
        h = F.celu(h)
        h = F.dropout(h, p=0.35, training=self.training)
        h = self.gat2(h, edge_index)
        h = F.dropout(h, p=0.15, training=self.training)
        x = pyg.nn.global_mean_pool(h, batch)
        x = self.head(x)
        
        return x

    def fit(self, train_loader, val_loader, epochs):
        self.to(device)  # Mueve el modelo a la GPU si está disponible
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=0.01)
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

        self.train()
        num_train_batches = len(train_loader)
        num_val_batches = len(val_loader)

        for epoch in range(epochs):
            total_train_r2 = 0.0
            total_train_mse = 0.0
            total_train_mae = 0.0

            for train_data in train_loader:
                train_data = train_data.to(device)
                train_out = self(train_data.x, train_data.edge_index, train_data.batch)
                train_loss = criterion(train_out.view(-1), train_data.y[:, self.target_index])

                train_r2 = r2score(train_out.view(-1), train_data.y[:, self.target_index])
                train_mse = mse_metric(train_out.view(-1), train_data.y[:, self.target_index])
                train_mae = mae_metric(train_out.view(-1), train_data.y[:, self.target_index])

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
                    val_out = self(val_data.x, val_data.edge_index, val_data.batch)
                    val_loss = criterion(val_out.view(-1), val_data.y[:, self.target_index])

                    val_r2 = r2score(val_out.view(-1), val_data.y[:, self.target_index])
                    val_mse = mse_metric(val_out.view(-1), val_data.y[:, self.target_index])
                    val_mae = mae_metric(val_out.view(-1), val_data.y[:, self.target_index])

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



    @torch.no_grad()
    def test(self, test_loader):
        criterion = nn.MSELoss()
        total_loss = 0
        mse_metric = MeanSquaredError()
        mae_metric = MeanAbsoluteError()
        r2score = R2Score()
       
        total_r2 = 0.0
        total_mse = 0.0
        total_mae = 0.0         
        self.eval()
        y_pred = []
        y_test = []        
        num_batches = len(test_loader)
        for data in test_loader:
            data = data.to(device)
            test_output = self(data.x, data.edge_index,data.batch)
            y_test.extend(data.y[:, self.target_index].cpu().tolist())
            y_pred.extend(test_output.view(-1).tolist())            
            loss = criterion(test_output.view(-1),data.y[:, self.target_index])
            r2 = r2score(test_output.view(-1).cpu(), data.y[:, self.target_index].cpu())
            mse = mse_metric(test_output.view(-1).cpu(), data.y[:, self.target_index].cpu())
            mae = mae_metric(test_output.view(-1).cpu(), data.y[:, self.target_index].cpu())                
            total_r2 += r2.item()
            total_mse += mse.item()
            total_mae += mae.item()  
            total_loss += loss.item() 
        average_loss = total_loss / num_batches
        mean_mse = total_mae / num_batches
        mean_mae = total_mse / num_batches
        mean_r2 = total_r2 / num_batches
        return y_pred,y_test,mean_mse, mean_mae, mean_r2,average_loss      
    def init_weights(self):
        # Inicializa los pesos de la red.
        for layer in self.modules():
            if isinstance(layer, (nn.Linear, GATv2Conv)):
                nn.init.xavier_uniform_(layer.weight.data)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)


    def shifted_softplus(self, x):         
        # Calcula la Shifted Softplus         
        shift = torch.log(torch.tensor(2.0)).item()         
        return F.softplus(x) - shift     
        
def main(args):
    #k_sizes = [3000, 5000]
    k_sizes = [3000, 5000, 10000, 30000, 50000, 100000, 130000]
    all_loss_values = []
    stats_all = []
    dataset = QM9(root="../../Dataset/QM9")
    for k in k_sizes:
        train_dataset, val_dataset, test_dataset = utils.split_dataset(dataset, k)
        train_loader, val_loader, test_loader = utils.create_data_loaders(train_dataset, val_dataset, test_dataset, args.batch_size)
        gat = GAT(dataset.num_node_features, args.hidden_size, 1, target_index=args.target_index)
        mean_train_mse, mean_train_mae, mean_train_r2, mean_val_mse, mean_val_mae, mean_val_r2, loss_values = gat.fit(train_loader, val_loader=val_loader, epochs=args.epochs)

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
        
        y_pred, y_test, mean_mse_t, mean_mae_t, mean_r2_t,average_loss = gat.test(test_loader)
        print(f"Test Loss:")
        print(f'Mean MSE: {mean_mse_t:.4f}')
        print(f'Mean MAE: {mean_mae_t:.4f}')
        print(f'Mean R²: {mean_r2_t:.4f}')

        stats = {'k': k, 'Mean MSE': mean_train_mse, 'Mean MAE': mean_train_mae, 'Mean R²': mean_train_r2,
                       'Validation Mean MSE': mean_val_mse, 'Validation Mean MAE': mean_val_mae,
                       'Validation Mean R²': mean_val_r2,
                       'Test Mean MSE': mean_mse_t, 'Test Mean MAE': mean_mae_t, 'Test Mean R²': mean_r2_t}
        stats_all.append(stats)
        utils.plot_scatter(y_test, y_pred, k=k,arq_name = 'GAT')

        residuals = [actual - predicted for actual, predicted in zip(y_test, y_pred)]
        utils.plot_residuals(y_pred, residuals, k=k,arq_name = 'GAT')

    # Gráfica de pérdida durante el entrenamiento
    utils.guardar_lista_diccionarios_a_csv(stats_all, 'GAT')
    utils.plot_loss_convergence(all_loss_values, list(map(str, k_sizes)), 'GAT')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenar una red GAT')
    parser.add_argument('--n', type=int, default=30000, help='Tamaño del conjunto de datos')
    parser.add_argument('--data_train_size', type=float, default=0.7, help='Proporción de datos de entrenamiento')
    parser.add_argument('--data_val_size', type=float, default=0.15, help='Proporción de datos de validacion')
    parser.add_argument('--data_test_size', type=float, default=0.30, help='Proporción de datos de prueba')
    parser.add_argument('--batch_size', type=int, default=16, help='Tamaño del lote (batch_size)')
    parser.add_argument('--hidden_size', type=int, default=128, help='Dimensión oculta (dim_h)')    
    parser.add_argument('--target_index', type=int, default= 6, help='target')
    parser.add_argument('--epochs', type=int, default= 12, help='Numero de epocas')
    args = parser.parse_args()
    main(args)
