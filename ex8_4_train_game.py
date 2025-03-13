import os
import ast
import torch
import numpy as np
from config import SIZE
from ex8_2_neural_network import GameNetwork

cnn_model = GameNetwork(SIZE)
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.01)


# Load model if it exists
if os.path.exists("game_network.pth"):
    cnn_model.load("game_network.pth")
    print("Model loaded successfully!")


# Read the training data file
training_data = []
with open("training_set1.txt", "r") as f:
    for line in f:
        try:
            entry = ast.literal_eval(line.strip())  # Read each entry separately
            training_data.append(entry)
        except Exception as e:
            print(f"Skipping corrupted line: {line}")

games_state = [entry[0] for entry in training_data]  # Game states
# print("X sample:", X[0])
policy_distribution = [entry[1][0] for entry in training_data]  # Policy distribution
print("P sum:", sum(policy_distribution[0]))  # אמור להיות 1
games_result = [entry[1][1] for entry in training_data]  # Game result (win/loss/draw)
print("Q sample:", games_result[0])  # צריך להיות ערך ריאלי, לא תמיד 0



games_state_array = np.array(games_state, dtype=np.float32)
policy_distribution_array = np.array(policy_distribution, dtype=np.float32)
games_result_array = np.array(games_result, dtype=np.float32).reshape(-1, 1)
 
games_state_tensor = torch.tensor(games_state_array, dtype=torch.float32) # Game states
games_state_tensor = games_state_tensor.view(-1, 3, SIZE, SIZE)  # Reshape to (batch_size, 3, 9, 9)

policy_distribution_tensor = torch.tensor(policy_distribution, dtype=torch.float32)  # Policy distribution

games_result_tensor = torch.tensor(games_result, dtype=torch.float32)  # Game result (win/loss/draw)
games_result_tensor = games_result_tensor.view(-1, 1)  # Reshape to (batch_size, 1)


# Print tensor shapes to confirm correctness
print(f"Loaded {len(training_data)} samples.")
print(f"X shape: {games_state_tensor.shape}")  # (num_samples, board_size, board_size)
print(f"P shape: {policy_distribution_tensor.shape}")  # (num_samples, board_size * board_size)
print(f"Q shape: {games_result_tensor.shape}")  # (num_samples, 1)


num_epochs = 4

# train the model
for epoch in range(num_epochs): 
    total_loss = 0
    for i in range(len(games_state_tensor)):  
        optimizer.zero_grad()
   
        # Forward pass
        P_pred, Q_pred = cnn_model(games_state_tensor[i].unsqueeze(0))  # Add batch dimension

        # Compute loss
        loss = cnn_model.compute_loss(P_pred, Q_pred, policy_distribution_tensor[i], games_result_tensor[i])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
 
        
    avg_loss = total_loss / len(training_data)
    print(f"Epoch {epoch+1}/{num_epochs}: Avg Loss = {avg_loss:.4f}")

# Save trained model
cnn_model.save("game_network.pth")
print("Training complete. Model saved!")

