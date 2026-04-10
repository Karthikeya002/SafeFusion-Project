import torch
import torch.nn as nn
import torch.optim as optim

# Dummy dataset
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

# Simple model
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

print("=====================================")
print("🚀 Transformer Training Started")
print("=====================================")

for epoch in range(epochs):

    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y).float().mean()

    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f} | Accuracy: {accuracy.item():.4f}")

print("=====================================")
print("✅ Training Completed")
print("=====================================")