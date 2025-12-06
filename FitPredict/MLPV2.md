Cell 1 — Imports & Setup (Code Cell)

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
```

Cell 2 — Load + Clean Data (Code Cell)
## 1. Data Loading and Preprocessing

```python
# Load CSV
df = pd.read_csv('gym_members_exercise_tracking.csv')
df.head()
```

2. Quick exploratory plots (on the raw data)

```python
# 2.1 Weight vs Height joint plot
plt.figure(figsize=(6, 4))
plt.scatter(df['Weight (kg)'], df['Height (m)'])
plt.title('Weight vs Height')
plt.xlabel('Weight (kg)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

# 2.2 Pairplot of numeric features for visual inspection
numeric_cols = df.select_dtypes(include=['number']).columns
df_explore = df[numeric_cols]

sns.pairplot(df_explore, height=2.5)
plt.tight_layout()
plt.show()
```

3.Handle missing values (on full dataframe)

```python
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns

# Numeric: fill with column means
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Categorical: fill with most frequent value (mode)
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

```

4. Feature Engineering

```python
# Feature Engineering
df['Workout_Intensity'] = df['Avg_BPM'] / df['Resting_BPM']
```

 5. Drop features based on pairplot inspection
   - low/no relationship with target (Calories_Burned)
   - or redundant because BMI already encodes height/weight

```python
drop_from_pairplot = [
    'Age',
    'Resting_BPM',
    'Water_Intake (liters)',
    'Workout_Frequency (days/week)',
    'Fat_Percentage',
    'Weight (kg)',
    'Height (m)'
]

# Only drop columns that actually exist (defensive)
drop_from_pairplot = [c for c in drop_from_pairplot if c in df.columns]

df_reduced = df.drop(columns=drop_from_pairplot)
print("After dropping low-value / redundant features:\n", df_reduced.columns, "\n")
```

 6. Correlation matrix on numeric features ONLY
    (temporarily drop categorical variables)

```python
df_corr = df_reduced.drop(columns=['Gender', 'Workout_Type'])

plt.figure(figsize=(12, 10))
corr_matrix = df_corr.corr()
sns.heatmap(
    corr_matrix,
    cbar=True,
    annot=False,
    square=True,
    cmap="coolwarm"
)
plt.title("Correlation Matrix (Numeric Features Only)")
plt.tight_layout()
plt.show()
```

 7. Final modelling dataframe:
 - ensure Experience_Level is numeric
 - one-hot encode categorical variables

```python
df_model = df_reduced.copy()

# If Experience_Level is categorical/ordinal, cast to float
df_model['Experience_Level'] = df_model['Experience_Level'].astype(float)

# One-hot encode Gender and Workout_Type
df_model = pd.get_dummies(df_model, columns=['Gender', 'Workout_Type'])

# This is the final dataset you’ll feed into Linear/Ridge/LASSO/MLP
df_model.head()
print("Final modelling columns:\n", df_model.columns)
```

8.Train/Test Split and Scaling

```python
target = 'Calories_Burned'

X = df_model.drop(columns=[target])
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

```

9.MLP Model Definition

```python
# Device and tensor conversion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1).to(device)

# Model definition
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

# Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 32
epochs = 100

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Lists for plotting
train_losses = []


# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)

    # Average loss this epoch
    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)



    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} ")


```

9.Plot Epoch Loss

```python
# Plot training loss
plt.figure(figsize=(8, 4))
plt.plot(range(1, epochs + 1), train_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Training Loss (MSE)')
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.show()




```

10.valuation (Code Cell)


```python
# Evaluation on test set
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).cpu().numpy().flatten()
    y_true = y_test_tensor.cpu().numpy().flatten()

mse = mean_squared_error(y_true, predictions)
r2 = r2_score(y_true, predictions)

print(f"MSE: {mse}")
print(f"R2: {r2}")




```

```python

```

