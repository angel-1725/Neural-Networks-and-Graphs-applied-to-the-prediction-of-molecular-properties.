import os
import csv

from rdkit import Chem


def read_xyz(file_name):
    with open(file_name, 'rb') as file:
        num_atoms = int(file.readline())
        properties = [float(num.replace(b'*^', b'e')) for num in file.readline().split()[1:17]]
        [file.readline() for _ in range(num_atoms)]
        vib_freqs = file.readline()
        smiles = file.readline().split()[0]
        inchis = file.readline()
    return smiles, properties



dir_path = "../../Dataset/dsgdb9nsd.xyz/"
mols = []
y = []

for filename in os.listdir(dir_path):
    if filename.endswith('.xyz'):
        file_path = os.path.join(dir_path, filename)
        smiles, properties = read_xyz(file_path)
        y.append(properties)
        mol = Chem.MolFromSmiles(smiles)
        mols.append(mol)

target = 5
y_target = [i[target]for i in y]
print(y_target)

nombre_archivo = 'y_target.csv'

# Guarda la lista en el archivo CSV
with open(nombre_archivo, 'w', newline='\n') as archivo_csv:
    escritor_csv = csv.writer(archivo_csv)
    for elemento in y_target:
        escritor_csv.writerow([elemento])
