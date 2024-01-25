import sys
sys.path.append('../Utilities') 
import utils
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.regression import R2Score  
from torchmetrics import MeanSquaredError, MeanAbsoluteError
HAR2EV = 27.211386246
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size=64):
        super(NeuralNetwork, self).__init__()
        self.batch_size = batch_size
        hidden_size_2 = int(hidden_size / 2)
        self.hidden_1 = nn.Linear(input_size, hidden_size)
        self.hidden_2 = nn.Linear(hidden_size, hidden_size)
        self.hidden_3 = nn.Linear(hidden_size, hidden_size_2)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size_2, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.00001)

    def forward(self, x):
        out = self.hidden_1(x)
        out = self.relu(out)
        out = self.hidden_2(out)
        out = self.relu(out)
        out = self.hidden_3(out)
        out = self.relu(out)
        out = self.output(out)
        return out

    def fit(self, X_train, y_train, X_val, y_val, num_epochs=1500):
        loss_values = []
        # Convertir tensores de validación a tipo BatchTensor para la evaluación en el bucle de entrenamiento
        X_val_batch = torch.cat([X_val] * (len(X_train) // len(X_val)))
        y_val_batch = torch.cat([y_val] * (len(y_train) // len(y_val)))
        
        mse_metric = MeanSquaredError()
        mae_metric = MeanAbsoluteError()
        r2score = R2Score()

        mean_train_mse = None
        mean_train_mae = None
        mean_train_r2 = None
        mean_val_mse = None
        mean_val_mae = None
        mean_val_r2 = None
        num_train_batches = len(X_val_batch)
        num_val_batches = len(y_val_batch)

        for epoch in range(num_epochs):
            # Iterar sobre lotes
            total_train_r2 = 0.0
            total_train_mse = 0.0
            total_train_mae = 0.0

            for i in range(0, len(X_train), self.batch_size):
                X_batch = X_train[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]

                train_out = self(X_batch)
                loss = self.criterion(train_out, y_batch)

                train_r2 = r2score(train_out, y_batch)
                train_mse = mse_metric(train_out, y_batch)
                train_mae = mae_metric(train_out, y_batch)
                total_train_r2 += train_r2.item()
                total_train_mse += train_mse.item()
                total_train_mae += train_mae.item()                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
  
            total_val_r2 = 0.0
            total_val_mse = 0.0
            total_val_mae = 0.0    
            with torch.no_grad():
                val_out = self(X_val_batch)
                val_loss = self.criterion(val_out, y_val_batch)
                val_r2 = r2score(val_out, y_val_batch)
                val_mse = mse_metric(val_out, y_val_batch)
                val_mae = mae_metric(val_out, y_val_batch)
                total_val_r2 += val_r2.item()
                total_val_mse += val_mse.item()
                total_val_mae += val_mae.item()


            if epoch % 50 == 0:
                loss_values.append(float(loss.item()))

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

            mean_train_mse = total_train_mse / num_train_batches
            mean_train_mae = total_train_mae / num_train_batches
            mean_train_r2 = total_train_r2 / num_train_batches

            mean_val_mse = total_val_mse / num_val_batches
            mean_val_mae = total_val_mae / num_val_batches
            mean_val_r2 = total_val_r2 / num_val_batches

            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, "
                  f"Train MSE: {mean_train_mse:.4f}, Val MSE: {mean_val_mse:.4f}, "
                  f"Train MAE: {mean_train_mae:.4f}, Val MAE: {mean_val_mae:.4f}, "
                  f"Train R2: {mean_train_r2:.4f}, Val R2: {mean_val_r2:.4f}")

        return mean_train_mse, mean_train_mae, mean_train_r2, mean_val_mse, mean_val_mae, mean_val_r2, loss_values

    @torch.no_grad()
    def test(self, X_test, y_test):
        # Iterar sobre lotes para la evaluación
        predictions = []
        total_loss = 0
        total_r2 = 0.0
        total_mse = 0.0
        total_mae = 0.0
        mse_metric = MeanSquaredError()
        mae_metric = MeanAbsoluteError()
        r2score = R2Score()
        num_batches = 0

        for i in range(0, len(X_test), self.batch_size):
            X_batch = X_test[i:i + self.batch_size]
            y_batch = y_test[i:i + self.batch_size]

            y_pred_batch = self(X_batch)
            loss = self.criterion(y_pred_batch, y_batch)
            
            r2 = r2score(y_pred_batch, y_batch)
            mse = mse_metric(y_pred_batch, y_batch)
            mae = mae_metric(y_pred_batch, y_batch)
            total_r2 += r2.item()
            total_mse += mse.item()
            total_mae += mae.item()                 

            predictions.append(y_pred_batch)

            num_batches += 1 
            total_loss += loss.item() 
        # Concatenar las predicciones y calcular la pérdida total
        y_pred = torch.cat(predictions)


        average_loss = total_loss / num_batches
        mean_mse = total_mae / num_batches
        mean_mae = total_mse / num_batches
        mean_r2 = total_r2 / num_batches

        return y_pred,y_test,mean_mse, mean_mae, mean_r2,average_loss   

def main():
    dir_path = "../../Dataset/dsgdb9nsd.xyz/"
    #k_sizes= [500, 1000]
    k_sizes= [3000, 5000,10000,30000,50000,100000,130000]
    all_loss_values = []
    all_stats = []
    dataset = utils.qm9_dataset(target=10)
    dataset.load_and_process_data(dir_path=dir_path,name_descriptors = 'rdkit')
    dataset.y = [i * HAR2EV for i in dataset.y]
    for k in k_sizes:
        X, y = dataset.get_sample(sample_size=k)
        X_tensor, y_tensor = utils.prepare_data_for_pytorch(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test = utils.split_data(X_tensor, y_tensor)

        input_size = X_train.shape[1]
        hidden_size = 256
        output_size = 1
        neural_network = NeuralNetwork(input_size, hidden_size, output_size)
        epochs = 1500
        mean_train_mse, mean_train_mae, mean_train_r2, mean_val_mse, mean_val_mae, mean_val_r2, loss_values = neural_network.fit(X_train, y_train, X_val, y_val, num_epochs = epochs)
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
        
        y_pred, y_test, mean_mse_t, mean_mae_t, mean_r2_t,average_loss = neural_network.test(X_test,y_test=y_test)
        print(f"Test Loss:")
        print(f'Mean MSE: {mean_mse_t:.4f}')
        print(f'Mean MAE: {mean_mae_t:.4f}')
        print(f'Mean R²: {mean_r2_t:.4f}')

        stats = {'k': k, 'Mean MSE': mean_train_mse, 'Mean MAE': mean_train_mae, 'Mean R²': mean_train_r2,
                       'Validation Mean MSE': mean_val_mse, 'Validation Mean MAE': mean_val_mae,
                       'Validation Mean R²': mean_val_r2,
                       'Test Mean MSE': mean_mse_t, 'Test Mean MAE': mean_mae_t, 'Test Mean R²': mean_r2_t}
        all_stats.append(stats)
        utils.plot_scatter(y_test, y_pred,k=k,arq_name='NN_rdkit')

        residuals = y_test - y_pred
        utils.plot_residuals(y_pred, residuals,k=k,arq_name='NN_rdkit')

        # Gráfica de pérdida durante el entrenamiento
    utils.guardar_lista_diccionarios_a_csv(all_stats,'NN_rdkit')
    utils.plot_loss_convergence(all_loss_values, list(map(str, k_sizes)),'NN_rdkit',save_interval= 50)

if __name__ == "__main__":
    main()
