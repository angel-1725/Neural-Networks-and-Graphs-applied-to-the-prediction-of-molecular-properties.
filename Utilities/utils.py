import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import Descriptors
import random
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm 
from torch_geometric.loader import DataLoader
def read_xyz(file_name):
    with open(file_name, 'rb') as file:
        num_atoms = int(file.readline())
        properties = [float(num.replace(b'*^', b'e')) for num in file.readline().split()[1:17]]
        [file.readline() for _ in range(num_atoms)]
        vib_freqs = file.readline()
        smiles = file.readline().split()[0]
        inchis = file.readline()
    return smiles, properties

def calculate_descriptors_with_rdkit(molecules):
    descriptors_list = []
    for mol in molecules:
        descriptors = {"SMILES": Chem.MolToSmiles(mol)}
        for descriptor_name, descriptor_function in Descriptors.descList:
            try:
                descriptor_value = descriptor_function(mol)
                descriptors[descriptor_name] = descriptor_value
            except:
                descriptors[descriptor_name] = None
        descriptors_list.append(descriptors)
    return descriptors_list
# Definición de la función llamada calculate_descriptors_with_mordred que toma una lista de moléculas como entrada
def calculate_descriptors_with_mordred(molecules):
    # Se crea una instancia de Calculator de Mordred y se habilita el cálculo de descriptores 3D
    calc = Calculator(descriptors, ignore_3D=False)
    #Calcula los descriptores para las moléculas utilizando el objeto Calculator
    descriptor_list = calc.pandas(molecules)
    #Devulve el DataFrame que contiene los descriptores calculados
    return descriptor_list
def remove_invalid_columns(descriptors):
    columns_to_drop = [col for col in descriptors.columns if not descriptors[col].apply(lambda x: isinstance(x, (int, float))).all()]
    descriptors = descriptors.drop(columns=columns_to_drop, errors='ignore')
    return descriptors

def remove_correlated_features(descriptors, threshold=0.9):
    print('correlated_matrix')
    correlated_matrix = descriptors.corr().abs()
    print('upper_triangle')
    upper_triangle = correlated_matrix.where(np.triu(np.ones(correlated_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in tqdm(upper_triangle.columns) if any(upper_triangle[column] >= threshold)]
    descriptors_correlated_dropped = descriptors.drop(columns=to_drop)
    return descriptors_correlated_dropped
def remove_low_variance(input_data, threshold=0.1):
    variances = input_data.var()
    selected_features = variances[variances >= threshold].index.tolist()
    filtered_data = input_data[selected_features]

    return filtered_data

def prepare_data_for_pytorch(X, y):
    y_tensor = torch.tensor(y)
    X_tensor = torch.from_numpy(X.astype(float))
    X_tensor = X_tensor.float()
    y_tensor = y_tensor.float().view(-1, 1)
    return X_tensor, y_tensor

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    """
    Crea y devuelve DataLoaders para conjuntos de entrenamiento, validación y prueba.

    Parameters:
    - train_dataset: Conjunto de datos de entrenamiento.
    - val_dataset: Conjunto de datos de validación.
    - test_dataset: Conjunto de datos de prueba.
    - batch_size: Tamaño del lote para los DataLoaders.

    Returns:
    - train_loader: DataLoader para el conjunto de entrenamiento.
    - val_loader: DataLoader para el conjunto de validación.
    - test_loader: DataLoader para el conjunto de prueba.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def split_dataset(dataset, n, data_train_size=0.6, data_val_size=0.2, data_test_size=0.2):
    """
    Divide un dataset en conjuntos de entrenamiento, validación y prueba.

    Parameters:
    - dataset: La dataset a dividir.
    - n: La cantidad total de datos a seleccionar aleatoriamente.
    - data_train_size: Proporción del dataset para entrenamiento (por defecto 0.6).
    - data_val_size: Proporción del dataset para validación (por defecto 0.2).
    - data_test_size: Proporción del dataset para prueba (por defecto 0.2).

    Returns:
    - train_dataset: Conjunto de datos de entrenamiento.
    - val_dataset: Conjunto de datos de validación.
    - test_dataset: Conjunto de datos de prueba.
    """
    index_train, index_val, index_test = calculate_indices(n, data_train_size, data_val_size, data_test_size)

    random_indices = random.sample(range(len(dataset)), n)

    train_indices = random_indices[:index_train]
    val_indices = random_indices[index_train:index_train + index_val]
    test_indices = random_indices[index_train + index_val:index_train + index_val + index_test]

    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    test_dataset = dataset[test_indices]

    return train_dataset, val_dataset, test_dataset

def calculate_indices(n, data_train_size, data_val_size, data_test_size):
    """
    Calcula los índices de los conjuntos de entrenamiento, validación y prueba.

    Parameters:
    - n: La cantidad total de datos a seleccionar aleatoriamente.
    - data_train_size: Proporción del dataset para entrenamiento.
    - data_val_size: Proporción del dataset para validación.
    - data_test_size: Proporción del dataset para prueba.

    Returns:
    - index_train: Índice para el conjunto de entrenamiento.
    - index_val: Índice para el conjunto de validación.
    - index_test: Índice para el conjunto de prueba.
    """
    index_train = int(data_train_size * n)
    index_val = int(data_val_size * n)
    index_test = int(data_test_size * n)

    return index_train, index_val, index_test


def plot_scatter(y_test:list, y_pred:list, k:int,arq_name:str):
    min_abs = min(y_test) if min(y_test)<=min(y_pred) else min(y_pred)
    max_abs = max(y_test) if max(y_test)>=max(y_pred) else max(y_pred)
    min_abs = min_abs
    max_abs = max_abs
    plt.xlim(min_abs, max_abs)
    plt.ylim(min_abs, max_abs)    
    plt.scatter(y_test, y_pred)
    plt.plot([min_abs, max_abs], [min_abs, max_abs], 'r--')
    plt.xlabel("Valores Verdaderos")
    plt.ylabel("Valores Predichos")
    plt.title(f'Dispersión: Real vs. Predicho. Zero point vibrational energy. k = {k}')
    plt.axis('equal')
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f'../../Img/dispersion_{str(k)}_{arq_name}.png')
    plt.close()


def plot_residuals(y_pred, residuals,k,arq_name:str):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Valores Predichos")
    plt.ylabel("Residuos")
    plt.title(f'Residuos. Zero point vibrational energy. k = {k}')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(f'../../Img/residuals_{str(k)}_{arq_name}.png')
    plt.close()
def plot_loss_convergence(all_loss_values, dataset_labels, arq_name:str,save_interval=1):
    plt.figure(figsize=(10, 6))

    for i, loss_values in enumerate(all_loss_values):
        epochs_to_plot = list(range(0, len(loss_values) * save_interval, save_interval))
        plt.plot(epochs_to_plot, loss_values, label=f'{dataset_labels[i]}', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Convergence for Different Datasets')
    plt.legend()
    plt.savefig(f'../../Img/convergence_{arq_name}.png')
    plt.close()


def guardar_lista_diccionarios_a_csv(lista_diccionarios, arq_name):
    if not lista_diccionarios:
        print("La lista de diccionarios está vacía.")
        return

    # Obtener las claves (encabezados) del primer diccionario en la lista
    encabezados = lista_diccionarios[0].keys()
    nombre_archivo = f'../../Stats/stats_{arq_name}.csv'
    # Abrir el archivo CSV en modo de escritura
    with open(nombre_archivo, 'w', newline='') as archivo_csv:
        # Crear un objeto DictWriter
        escritor_csv = csv.DictWriter(archivo_csv, fieldnames=encabezados)

        # Escribir los encabezados en el archivo
        escritor_csv.writeheader()

        # Escribir cada diccionario en la lista como una fila en el archivo
        for diccionario in lista_diccionarios:
            escritor_csv.writerow(diccionario)

    print(f"La lista de diccionarios ha sido guardada en {nombre_archivo}.")



import matplotlib.pyplot as plt

def guardar_grafica_lineas_multiples_2(metrica_modelo1, metrica_modelo2, metrica_modelo3, tamanos_k, nombre_metrica, nombre_modelo1='Modelo 1', nombre_modelo2='Modelo 2', nombre_modelo3='Modelo 3', nombre_archivo='fig.png', direccion='.'):
    """
    Guarda una gráfica de líneas múltiples con tres métricas de modelos diferentes en un archivo.

    Parámetros:
    - direccion: La dirección donde se guardará la figura.
    - tamanos_k: Un arreglo con los tamaños de k que serán los labels del eje x.
    - metrica_modelo1: Lista de valores de la métrica para el modelo 1.
    - metrica_modelo2: Lista de valores de la métrica para el modelo 2.
    - metrica_modelo3: Lista de valores de la métrica para el modelo 3.
    - nombre_metrica: Nombre de la métrica para etiquetar el eje y y el título de la gráfica.
    - nombre_modelo1: Nombre del modelo 1 para la leyenda de la gráfica.
    - nombre_modelo2: Nombre del modelo 2 para la leyenda de la gráfica.
    - nombre_modelo3: Nombre del modelo 3 para la leyenda de la gráfica.
    - nombre_archivo: El nombre del archivo de la figura a guardar.
    """
    plt.figure(figsize=(10, 6))

    # Graficar las métricas para cada modelo
    plt.plot(tamanos_k, metrica_modelo1, label=nombre_modelo1, marker='o')
    plt.plot(tamanos_k, metrica_modelo2, label=nombre_modelo2, marker='o')
    plt.plot(tamanos_k, metrica_modelo3, label=nombre_modelo3, marker='o')

    # Configuración de la gráfica
    plt.xlabel('Tamaño de k')
    plt.ylabel(nombre_metrica)
    plt.title(f'Comparación de {nombre_metrica} para {nombre_modelo1}, {nombre_modelo2} y {nombre_modelo3}')
    plt.legend()
    plt.grid(True)

    # Guardar la figura en la dirección especificada
    ruta_completa = f"{direccion}/{nombre_archivo}"
    plt.savefig(ruta_completa)





def guardar_grafica_lineas_multiples(metrica_modelo1, metrica_modelo2, tamanos_k, nombre_metrica, nombre_modelo1='Modelo 1', nombre_modelo2='Modelo 2', nombre_archivo='fig.png', direccion='.'):
    """
    Guarda una gráfica de líneas múltiples con dos métricas de modelos diferentes en un archivo.

    Parámetros:
    - direccion: La dirección donde se guardará la figura.
    - tamanos_k: Un arreglo con los tamaños de k que serán los labels del eje x.
    - metrica_modelo1: Lista de valores de la métrica para el modelo 1.
    - metrica_modelo2: Lista de valores de la métrica para el modelo 2.
    - nombre_metrica: Nombre de la métrica para etiquetar el eje y y el título de la gráfica.
    - nombre_modelo1: Nombre del modelo 1 para la leyenda de la gráfica.
    - nombre_modelo2: Nombre del modelo 2 para la leyenda de la gráfica.
    - nombre_archivo: El nombre del archivo de la figura a guardar.
    """
    plt.figure(figsize=(10, 6))

    # Graficar las métricas para cada modelo
    plt.plot(tamanos_k, metrica_modelo1, label=nombre_modelo1, marker='o')
    plt.plot(tamanos_k, metrica_modelo2, label=nombre_modelo2, marker='o')

    # Configuración de la gráfica
    plt.xlabel('Tamaño de k')
    plt.ylabel(nombre_metrica)
    plt.title(f'Comparación de {nombre_metrica} para {nombre_modelo1} y {nombre_modelo2}')
    plt.legend()
    plt.grid(True)

    # Guardar la figura en la dirección especificada
    ruta_completa = f"{direccion}/{nombre_archivo}"
    plt.savefig(ruta_completa)

# Ejemplo de uso:
# guardar_grafica_lineas_multiples(metrica_valores_modelo1, metrica_valores_modelo2, tamanos_k, 'R²', 'Red Neuronal', 'Random Forest', 'fig.png', '.')


def grafica_cajas_comparativa(metrica_modelo1, metrica_modelo2, nombres_modelos, nombres_metricas, nombre_archivo='boxplot_nn.png', direccion='.'):
    """
    Guarda una gráfica de cajas comparativa para métricas de dos modelos.

    Parámetros:
    - direccion: La dirección donde se guardará la figura.
    - metrica_modelo1: Lista de valores de métricas para el modelo 1.
    - metrica_modelo2: Lista de valores de métricas para el modelo 2.
    - nombres_modelos: Lista de nombres de los modelos (por ejemplo, ['Modelo A', 'Modelo B']).
    - nombres_metricas: Lista de nombres de las métricas (por ejemplo, ['R^2', 'MSE', 'MAE']).
    - nombre_archivo: El nombre del archivo de la figura a guardar.
    """
    plt.figure(figsize=(10, 6))

    # Graficar cajas y bigotes para cada métrica y modelo
    plt.boxplot([metrica_modelo1, metrica_modelo2], labels=nombres_modelos)

    # Configuración de la gráfica
    plt.ylabel('Valor de la métrica')
    plt.title('Comparación de Métricas para Modelos')
    plt.xticks(rotation=45)
    plt.grid(True)

    # Añadir nombres de las métricas en el eje y
    plt.yticks([], [])
    for i, nombre_metrica in enumerate(nombres_metricas):
        plt.text(1.1, metrica_modelo2[i], nombre_metrica, va='center')

    # Guardar la figura en la dirección especificada
    ruta_completa = f"{direccion}/{nombre_archivo}"
    plt.savefig(ruta_completa)

def leer_csv_y_extraer_informacion(nombre_archivo):
    """
    Lee un archivo CSV y extrae información de las columnas.

    Parámetros:
    - nombre_archivo: Nombre del archivo CSV.

    Retorna:
    - Un diccionario con arreglos, donde la clave es el nombre de la columna y el valor es el arreglo de datos.
    """
    try:
        # Leer el archivo CSV con pandas
        df = pd.read_csv(nombre_archivo)

        # Extraer la información de la primera columna (suponiendo que la primera columna es 'k')
        k_sizes = df.iloc[:, 0].values

        # Extraer la información de las columnas restantes
        datos_por_columna = {}
        for columna in df.columns[1:]:
            datos_por_columna[columna] = df[columna].values

        # Almacenar la información en un diccionario
        informacion = {'k_sizes': k_sizes, **datos_por_columna}

        return informacion

    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return None


def prepare_data_for_pytorch(X, y):
    y_tensor = torch.tensor(y)
    X_tensor = torch.from_numpy(X.astype(float))
    X_tensor = X_tensor.float()
    y_tensor = y_tensor.float().view(-1, 1)
    return X_tensor, y_tensor
class qm9_dataset:
    def __init__(self,target):
        self.target = target
        self.X = None
        self.y = None

    def load_and_process_data(self, dir_path,name_descriptors):
        mols, y = [], []
        for filename in os.listdir(dir_path):
            if filename.endswith('.xyz'):
                file_path = os.path.join(dir_path, filename)
                smiles, properties = read_xyz(file_path)
                y.append(properties)
                mol = Chem.MolFromSmiles(smiles)
                mols.append(mol)

        self.y = [fila[self.target ] for fila in y]
        if name_descriptors == 'mordred':
            descriptors = calculate_descriptors_with_mordred(mols)
        elif name_descriptors == 'rdkit':
            descriptors = calculate_descriptors_with_rdkit(mols)
        
        pandas_df = pd.DataFrame(descriptors)
        pandas_df = remove_invalid_columns(pandas_df)
        self.X = remove_correlated_features(pandas_df)
        self.X = remove_low_variance(self.X, threshold=0.1)
    def get_sample(self, sample_size):
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Call load_and_process_data() first.")
        
        # Mezclar los datos y seleccionar una muestra.
        indices = np.random.choice(len(self.X), size=sample_size, replace=False)
        X_sample = pd.DataFrame(self.X).iloc[indices].values
        y_sample = pd.DataFrame(self.y).iloc[indices].values.flatten()
        
        return X_sample, y_sample            

