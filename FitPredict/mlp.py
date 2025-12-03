import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import TensorDataset, DataLoader

# 1. Data Handling
# Load the CSV dataset
df = pd.read_csv('../gym_members_exercise_tracking.csv')

# Missing Values
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encoding
df = pd.get_dummies(df, columns=['Gender', 'Workout_Type'])

# Experience_Level is already numeric (ordinal), ensure float
df['Experience_Level'] = df['Experience_Level'].astype(float)

# Feature Engineering
df['Workout_Intensity'] = df['Avg_BPM'] / df['Resting_BPM']

# Prepare X and y
target = 'Calories_Burned'
X = df.drop(columns=[target])
y = df[target]

# Train/test split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1).to(device)

# 2. Model Architecture
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

input_dim = X_train.shape[1]
model = MLP(input_dim).to(device)

# 3. Training Requirements
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 32
epochs = 100

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# 4. Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).cpu().numpy()
    y_true = y_test_tensor.cpu().numpy()

mse = mean_squared_error(y_true, predictions)
r2 = r2_score(y_true, predictions)

print(f"MSE: {mse}")
print(f"R2: {r2}")
