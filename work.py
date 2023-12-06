import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score
import json
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

# 读取数据
train_data = pd.read_csv("one2one_training1.csv")
train_data['winner'] = train_data['winner'].map({'p1': 0, 'p2': 1, 'DRAW': 2})
train_answer_tensor = torch.tensor(train_data['winner'].values)

valid_data = pd.read_csv("one2one_valid_query.csv")
valid_answer = pd.read_csv("one2one_valid_answer.csv")
valid_answer['winner'] = valid_answer['winner'].map({'p1': 0, 'p2': 1, 'DRAW': 2})
valid_answer_tensor = torch.tensor(valid_answer['winner'].values)

test_data = pd.read_csv("one2one_test_query.csv")


# 构建图数据集
class MyDataset(InMemoryDataset):
    def __init__(self, root, data_frame, transform=None, pre_transform=None):
        self.data_frame = data_frame
        super(MyDataset, self).__init__(root, transform, pre_transform)
        self.date_encoder = LabelEncoder()
        self.date_encoder.fit(self.data_frame['date'])
        self.player_encoder = LabelEncoder()
        players = pd.concat([self.data_frame['player 1'], self.data_frame['player 2']])
        self.player_encoder.fit(players)

    def process(self):
        data_list = []
        for idx, row in self.data_frame.iterrows():
            player1 = self.player_encoder.transform([row['player 1']])[0]
            player2 = self.player_encoder.transform([row['player 2']])[0]
            edge_index = torch.tensor([[player1, player2], [player2, player1]], dtype=torch.long)
            x = torch.eye(len(self.player_encoder.classes_))  # One-hot encoding of nodes
            edge_attr = torch.tensor(self.date_encoder.transform([row['date'], row['date']]), dtype=torch.long)  # Edge feature is the date
            y = torch.tensor(0 if row['winner'] == row['player 1'] else 1 if row['winner'] == row['player 2'] else 2)  # Label is the winner
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(data)

        self.data_list = data_list  # Assign data_list to self.data_list

    def len(self):
        return len(self.data_frame)

    def get(self, idx):
        return self.data_list[idx]

    def processed_file_names(self):
        return ['data.pt']

# 定义模型
class DualAttentionGNN(nn.Module):
    def __init__(self, num_players, num_dates, embedding_dim):
        super(DualAttentionGNN, self).__init__()

        self.emb_player = nn.Embedding(num_players, embedding_dim)
        self.emb_ts = nn.Embedding(num_dates, embedding_dim)

        self.conv1 = GCNConv(16, 16)
        self.conv2 = GCNConv(16, 16)
        self.conv3 = GCNConv(16, num_players)

        self.attn_in1 = nn.Parameter(torch.Tensor(1, 16))
        self.attn_out1 = nn.Parameter(torch.Tensor(1, 16))

        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x, edge_index, edge_attr, batch):
        input_transform = nn.Linear(x.size(1), 16).to(x.device)
        x = input_transform(x)
        x = F.relu(self.conv1(x, edge_index))

        ts_emb = self.emb_ts((edge_attr.argmax() % self.emb_ts.num_embeddings).unsqueeze(0))
        player_emb = self.emb_player((x.argmax(dim=1) % self.emb_player.num_embeddings).unsqueeze(1))

        ts_emb = ts_emb.squeeze(1)
        player_emb = player_emb.squeeze(1)

        ts_emb = ts_emb.expand(x.size(0), -1)  # Repeat ts_emb for each input in the batch
        player_emb = player_emb.expand(x.size(0), -1)  # Repeat player_emb for each input in the batch
        x = torch.cat([x, ts_emb, player_emb], dim=1)

        x_transform = nn.Linear(x.size(1), 16).to(x.device)
        x = x_transform(x)

        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        input_transform = nn.Linear(x.size(1), 16).to(x.device)
        x = input_transform(x)

        attn_in = F.relu((x * self.attn_in1).sum(dim=-1))
        attn_out = F.relu((x * self.attn_out1).sum(dim=-1))
        attn = F.softmax(torch.cat([attn_in, attn_out], dim=0), dim=0)

        x_transform = nn.Linear(x.size(1), attn.size(0)).to(x.device)
        x = x_transform(x)

        x = attn * x

        x = global_mean_pool(x, batch).to(x.device)  # Global mean pooling

        x_transform = nn.Linear(x.size(1), 16).to(x.device)
        x = x_transform(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x 

def train(loader, answer_tensor):
    model.train()
    total_loss = 0
    for i, data in tqdm(enumerate(loader)):  # Use enumerate to get i and data
        data = data.to(device)  # Move data to the correct device
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)  # Pass data.batch to the model
        
        # Get the corresponding slice of answer_tensor
        answer_slice = answer_tensor[i * loader.batch_size : (i + 1) * loader.batch_size].to(device)
        out = out.float()  # Convert to Float tensor
        answer_slice = answer_slice.long()  # Convert to Long tensor
        # print('answer:',answer_slice.min(), answer_slice.max())  # Check the range of values in answer_slice
        # print('out:',out.shape)  # Check the shape of out
        loss = criterion(out, answer_slice)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(loader, valid_answer_tensor):
    model.eval()
    correct = 0
    for i, data in tqdm(enumerate(loader)):
        with torch.no_grad():
            data = data.to(device) # Move data to the correct device
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)  # Use data.edge_attr
            answer_slice = valid_answer_tensor[i * loader.batch_size : (i + 1) * loader.batch_size].to(device)
            out = out.float()  # Convert to Float tensor
            answer_slice = answer_slice.long()  # Convert to Long tensor
            pred = out.argmax(dim=-1)  # Use dim=-1 to avoid errors if out is 1D
            correct += pred.eq(answer_slice).sum().item()
    return correct / len(loader.dataset)

def predict(loader):
    model.eval()
    preds = []
    for i, data in tqdm(enumerate(loader)):
        data = data.to(device)  # Move data to the correct device
        with torch.no_grad():  
            out = model(data.x, data.edge_index, data.edge_attr, data.batch) 
            preds.append(out.argmax(dim=-1))  # Use dim=-1 to avoid errors if out is 1D
    preds = torch.cat(preds).tolist()
    # Reverse mapping
    preds = ['p1' if pred == 0 else 'p2' if pred == 1 else 'DRAW' for pred in preds]
    return preds


# train eval test
from torch_geometric.data import DataLoader

root = './'
train_dataset = MyDataset(root, train_data)
valid_dataset = MyDataset(root, valid_data)
test_dataset = MyDataset(root, test_data)
train_dataset.process()
valid_dataset.process()
test_dataset.process()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
criterion = nn.CrossEntropyLoss()
#dataset = MyDataset(root, data_frame)
num_players = len(train_dataset.player_encoder.classes_)
num_dates = len(train_dataset.date_encoder.classes_)
embedding_dim = 16  # or any other value you want
model = DualAttentionGNN(num_players, num_dates, embedding_dim).to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 


#train
for epoch in range(1, 3):
    train_loss = train(train_loader, train_answer_tensor)
    val_acc = test(valid_loader, valid_answer_tensor)
    print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}')

# eval test
val_answers = torch.from_numpy(pd.to_numeric(valid_answer['winner'], errors='coerce').values)
val_acc = test(valid_loader, valid_answer_tensor)
test_preds = predict(test_dataset)

# output to csv file
submission = pd.DataFrame()
submission['game'] = test_data['game']
submission['winner'] = test_preds
submission.to_csv('submission.csv', index=False)