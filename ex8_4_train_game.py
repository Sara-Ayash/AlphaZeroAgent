import torch
import torch.optim as optim
from ex8_2_neural_network import GameNetwork
from ex8_3_puct import PUCTNode, PUCTPlayer
from config import SIZE,EMPTY, STRIKE, PLAYER_BLACK, PLAYER_WHITE, DRAW, ITERATIONS_NUMBER
import ast
import os

cnn_model = GameNetwork(SIZE)
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.01)


# Load model if it exists
if os.path.exists("game_network.pth"):
    cnn_model.load("game_network.pth")
    print("Model loaded successfully!")

# Read the training data file
training_data = []
with open("training_set.txt", "r") as f:
    for line in f:
        try:
            entry = ast.literal_eval(line.strip())  # Read each entry separately
            training_data.append(entry)
        except Exception as e:
            print(f"Skipping corrupted line: {line}")

X = [entry[0] for entry in training_data]  # Game states
# print("X sample:", X[0])
P = [entry[1][0] for entry in training_data]  # Policy distribution
print("P sum:", sum(P[0]))  # אמור להיות 1
Q = [entry[1][1] for entry in training_data]  # Game result (win/loss/draw)
print("Q sample:", Q[0])  # צריך להיות ערך ריאלי, לא תמיד 0



# Extract states (X), policies (P), and values (Q)
X = torch.tensor([entry[0] for entry in training_data], dtype=torch.float32)  # Game states
# print("X shape before view:", X.shape)
X = X.view(-1, 3, SIZE, SIZE)  # Reshape to (batch_size, 3, 9, 9)
# print("X shape after view:", X.shape)
# print("sample in X after view:\n", X[10])  # הדפסת דוגמה 

P = torch.tensor([entry[1][0] for entry in training_data], dtype=torch.float32)  # Policy distribution
# print("sample in P:\n", P[10])  # הדפסת דוגמה 

Q = torch.tensor([entry[1][1] for entry in training_data], dtype=torch.float32)  # Game result (win/loss/draw)
Q = Q.view(-1, 1)  # Reshape to (batch_size, 1)
# print("sample in Q after view:\n", Q[10])  # הדפסת דוגמה 


# Print tensor shapes to confirm correctness
print(f"Loaded {len(training_data)} samples.")
print(f"X shape: {X.shape}")  # (num_samples, board_size, board_size)
print(f"P shape: {P.shape}")  # (num_samples, board_size * board_size)
print(f"Q shape: {Q.shape}")  # (num_samples, 1)


num_epochs = 4

# train the model
for epoch in range(num_epochs): 
    total_loss = 0
    for i in range(len(X)):  
        optimizer.zero_grad()
   
        # Forward pass
        P_pred, Q_pred = cnn_model(X[i].unsqueeze(0))  # Add batch dimension

        # Compute loss
        loss = cnn_model.compute_loss(P_pred, Q_pred, P[i], Q[i])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
 
        
        

        
    avg_loss = total_loss / len(training_data)
    # print(f"Epoch {epoch+1}/{num_epochs}: Avg Loss = {avg_loss:.4f}")

    print(f"Epoch {epoch+1}/{num_epochs}: Avg Loss = {avg_loss:.4f}")
    print(f"Sample P_pred: {P_pred.detach().cpu().numpy()[:5]}")
    print(f"Sample Q_pred: {Q_pred.detach().cpu().numpy()[:5]}")
    print(f"Sample P_target: {P[i].cpu().numpy()[:5]}")
    print(f"Sample Q_target: {Q[i].cpu().numpy()}")



for param in cnn_model.parameters():
    print(param.grad)  # צריך להיות ערכים שונים מאפס


# Save trained model
cnn_model.save("game_network.pth")
print("Training complete. Model saved!")


# # Load trained model for testing
# cnn_model.load("game_network.pth")
# print("Model loaded successfully! Testing a prediction...")

# # Test with a sample from training data
# test_sample = X[0].unsqueeze(0)  # Add batch dimension
# P_pred, Q_pred = cnn_model(test_sample)

# print("Test Prediction:")
# print(f"P_pred: {P_pred}")
# print(f"Q_pred: {Q_pred}")



# cnn_model.load("game_network.pth")
# test_sample = X_tensor[0].unsqueeze(0)
# P_pred, Q_pred = cnn_model(test_sample)
# print("Model loaded successfully! Test prediction:")
# print(f"P_pred: {P_pred}, Q_pred: {Q_pred}")
