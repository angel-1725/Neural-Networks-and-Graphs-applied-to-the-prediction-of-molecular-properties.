# Neural-Networks-and-Graphs-applied-to-the-prediction-of-molecular-properties.
Este repositorio alberga la implementación en PyTorch y PyTorch Geometric de redes neuronales y redes neuronales con grafos destinadas a predecir propiedades moleculares en compuestos orgánicos pequeños. La información molecular utilizada proviene de la base de datos QM9. Se han desarrollado dos enfoques distintos para abordar esta tarea.

En la primera aproximación, se utiliza descriptores moleculares como datos de entrada para una red neuronal. Por otro lado, la segunda estrategia se basa en el uso de redes neuronales gráficas, y se han implementado tres arquitecturas diferentes que incorporan pasos de mensajes distintos para mejorar la capacidad de la red para aprender interacciones espaciales y químicas entre los elementos constituyentes de las moléculas.

El repositorio está vinculado a un proyecto de tesis llevado a cabo en colaboración con Diego Ramirez Ramirez, con la asistencia de los asesores Diego Gonzalez y Roberto Bernal Marquez, en la UAM Cuajimalpa.

Este repositorio aloja cuatro implementaciones de redes neuronales, siendo tres de ellas Grafos Neuronales (GNNs). Cada conjunto de códigos se encarga de entrenar una red neuronal para la predicción de propiedades moleculares y presenta un conjunto de gráficas que visualizan los resultados de la evaluación del entrenamiento y las predicciones realizadas.

Para organizar el código de manera estructurada, se ha asignado a cada arquitectura de red neuronal su propio directorio, lo que facilita la comprensión y navegación en el repositorio. Además del código específico de cada arquitectura, se incluye un conjunto de utilidades en el archivo utils.py, ubicado en el directorio "Utilities". Este archivo contiene funciones auxiliares compartidas entre las distintas implementaciones. Dichas funciones desempeñan diversas tareas, como la lectura de la base de datos, el procesamiento de información molecular y la generación de gráficos que representan los resultados de los procesos de evaluación del entrenamiento y la predicción de las redes neuronales.

Requisitos y Dependencias

## Configuración del Entorno para el Proyecto

Este proyecto requiere la instalación de varias bibliotecas y paquetes para garantizar su correcto funcionamiento. A continuación, se proporciona una lista de las librerías necesarias y las instrucciones para su instalación.

### 1. Librerías de Python

Asegúrese de tener instalada una versión de Python compatible (preferiblemente Python 3.8). Puede descargar la última versión de Python desde [python.org](https://www.python.org/downloads/).

### 2. Bibliotecas Esenciales

Instale las siguientes bibliotecas utilizando el gestor de paquetes `pip` en la terminal:


\`pip install numpy pandas matplotlib mordred rdkit scikit-learn tqdm torch\`
### 3. Torch Geometric

Para la manipulación de datos relacionados con grafos, es necesario instalar la biblioteca torch_geometric. Puede seguir las instrucciones detalladas en la página oficial de PyTorch Geometric para la instalación según su sistema operativo y configuración.
### 4. Torch Metrics

Para métricas específicas de PyTorch, como Mean Absolute Error (MAE), Mean Squared Error (MSE), y R2 Score, instale la biblioteca torchmetrics mediante:
pip install torchmetrics
### 5. Mordred y RDKit

Para el cálculo de descriptores moleculares, instale mordred y rdkit. Puede instalarlos utilizando pip:
pip install mordred-py
conda install -c conda-forge rdkit

Después de instalar estas librerías, puede ejecutar el código proporcionado en el proyecto sin problemas.

## Estructura del Proyecto
El repositorio está organizado en varias carpetas, cada una con un propósito específico:

    GAT:
        Contiene un archivo:
            gat.py: Implementa una Red Neuronal con arquitectura GAT (Graph Attention Network).

    GCN:
        Contiene un archivo:
            gcn.py: Implementa una Red Neuronal con arquitectura GCN (Graph Convolutional Network).

    NN_descriptors:
        Contiene dos archivos:
            NN_mordred.py: Implementa una Red Neuronal que utiliza descriptores moleculares obtenidos con la librería Mordred.
            NN_rdkit.py: Implementa una Red Neuronal que utiliza descriptores moleculares obtenidos con la librería RDKit.

    Schnet:
        Contiene un archivo:
            SchNet.py: Implementa una Red Neuronal con arquitectura SchNet.

    Utilities:
        Contiene un archivo:
            utils.py: Proporciona funciones auxiliares compartidas entre las diferentes implementaciones, como la lectura de bases de datos, procesamiento de información molecular y generación de gráficos para la evaluación del entrenamiento y predicción de las redes neuronales.
