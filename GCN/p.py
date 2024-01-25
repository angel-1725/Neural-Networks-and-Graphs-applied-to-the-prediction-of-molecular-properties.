from torch_geometric.datasets import QM9 
dataset = QM9(root="../../Dataset/QM9")
print(dataset[0])