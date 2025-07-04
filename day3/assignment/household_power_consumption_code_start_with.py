# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# %%
# load data
df = pd.read_csv(r'D:\SummerCampProgram\aiSummerCamp2025-1\day3\assignment\data\household_power_consumption\household_power_consumption.csv', sep = ";")
df.head()

# %%
# check the data
df.info()

# %%
df['datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'])
df.drop(['Date', 'Time'], axis = 1, inplace = True)
# handle missing values
df.dropna(inplace = True)

# %%
print("Start Date: ", df['datetime'].min())
print("End Date: ", df['datetime'].max())

# %%
# split training and test sets
# the prediction and test collections are separated over time
train, test = df.loc[df['datetime'] <= '2009-12-31'], df.loc[df['datetime'] > '2009-12-31']

# %%
# data normalization
scaler = MinMaxScaler()
feature_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
train_scaled = scaler.fit_transform(train[feature_cols])
test_scaled = scaler.transform(test[feature_cols])
# %%
# split X and y
def create_sequences(data, seq_length=24):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length][0]  # 预测Global_active_power
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LEN = 24  # 以一天为单位
X_train, y_train = create_sequences(train_scaled, SEQ_LEN)
X_test, y_test = create_sequences(test_scaled, SEQ_LEN)
# %%
# creat dataloaders
class PowerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

BATCH_SIZE = 64
train_dataset = PowerDataset(X_train, y_train)
test_dataset = PowerDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# %%
# build a LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.fc(out)
        return out.squeeze()

INPUT_SIZE = len(feature_cols)
HIDDEN_SIZE = 64
model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE)
# %%
# train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")
# %%
# evaluate the model on the test set
model.eval()
preds, trues = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        output = model(X_batch)
        preds.append(output.cpu().numpy())
        trues.append(y_batch.numpy())
preds = np.concatenate(preds)
trues = np.concatenate(trues)

# 反归一化
preds_inv = scaler.inverse_transform(
    np.concatenate([preds.reshape(-1,1), np.zeros((len(preds), len(feature_cols)-1))], axis=1)
)[:,0]
trues_inv = scaler.inverse_transform(
    np.concatenate([trues.reshape(-1,1), np.zeros((len(trues), len(feature_cols)-1))], axis=1)
)[:,0]

from sklearn.metrics import mean_squared_error
print("Test RMSE:", np.sqrt(mean_squared_error(trues_inv, preds_inv)))
# %%
# plotting the predictions against the ground truth
plt.figure(figsize=(15,5))
plt.plot(trues_inv[:500], label='True')
plt.plot(preds_inv[:500], label='Predicted')
plt.legend()
plt.title('Global Active Power Prediction')
plt.xlabel('Time Step')
plt.ylabel('Global Active Power')
plt.show()