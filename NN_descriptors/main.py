import os 
from rdkit import Chem
from rdkit.Chem import Descriptors
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


def read_xyz(file_name):
    with open(file_name, 'rb') as file:
        num_atoms = int(file.readline())
        properties = [float(num.replace(b'*^', b'e')) for num in file.readline().split()[1:17]]
        [file.readline() for _ in range(num_atoms)]
        vib_freqs = file.readline()
        smiles = file.readline().split()[0]
        inchis = file.readline()
    return smiles, properties


dir_path = '/home/angel_23/Documentos/PT/Codigos/dsgdb9nsd.xyz/'

mols, y = [], []
for i,filename in zip(range(1),os.listdir(dir_path)):
    if filename.endswith('.xyz'):
        file_path = os.path.join(dir_path, filename)
        smiles, properties = read_xyz(file_path)
        y.append(properties)
        mol = Chem.MolFromSmiles(smiles)
        mols.append(mol)

y = [fila[10 ] for fila in y]
descriptors = calculate_descriptors_with_rdkit(mols)

len(descriptors)