import sys
sys.path.append('../Utilities') 
import utils
import argparse
import os.path as osp

import torch
from tqdm import tqdm

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet

parser = argparse.ArgumentParser()
parser.add_argument('--cutoff', type=float, default=10.0,
                    help='Cutoff distance for interatomic interactions')
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..','Dataset', 'QM9')
dataset = QM9(path)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

target = 5
model, datasets = SchNet.from_qm9_pretrained(path, dataset, target)
train_dataset, val_dataset, test_dataset = datasets

model = model.to(device)
loader = DataLoader(test_dataset, batch_size=256)

maes = []
y_pred = []
y_test = []
for data in tqdm(loader):
    data = data.to(device)
    with torch.no_grad():
        pred = model(data.z, data.pos, data.batch)
    y_pred.extend(pred.view(-1).cpu().tolist())
    y_test.extend(data.y[:, target].cpu().tolist())
    mae = (pred.view(-1) - data.y[:, target]).abs()
    maes.append(mae)

mae = torch.cat(maes, dim=0)

# Report meV instead of eV.
mae = 1000 * mae if target in [2, 3, 4, 6, 7, 8, 9, 10] else mae

print(f'Target: {target:02d}, MAE: {mae.mean():.5f} ± {mae.std():.5f}')
utils.plot_scatter(y_test,y_pred,13000,'Schnet')