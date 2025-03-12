import torch
import torch.nn as nn
import torch.optim as optim
from config import SIZE,EMPTY, STRIKE, PLAYER_BLACK, PLAYER_WHITE, DRAW, ITERATIONS_NUMBER


class GameNetwork(nn.Module):
    def __init__(self, board_size=SIZE):
        """
        Neural network for predicting:
        - `P` (policy): probability of choosing each move
        - `Q` (value): estimated win probability

        Parameters:
        - board_size (int): The game board size
        """
        super(GameNetwork, self).__init__()
        
        self.board_size = board_size  # Store board size

        # Convolutional layers (extract spatial patterns)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * board_size * board_size, 256)
        self.fc2 = nn.Linear(256, 128)

        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()  # For policy
        self.value_loss_fn = nn.MSELoss()  # For value

        # Policy Head (P): Probabilities for moves
        self.policy_head = nn.Linear(128, board_size * board_size)

        # Value Head (Q): State evaluation (-1 to 1)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        """
        Forward pass.

        Parameters:
        - x: Encoded state (batch_size, 3, board_size, board_size)

        Returns:
        - policy: Action probabilities (softmax output)
        - value: Win probability (-1 to 1)
        """

        # Apply convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Flatten feature maps
        x = x.view(x.shape[0], -1)

        # Apply fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # Compute outputs
        policy = torch.softmax(self.policy_head(x), dim=-1)  # Probability for each move
        value = torch.tanh(self.value_head(x))  # Game state value

        return policy, value
    

    def compute_loss(self, P_pred, Q_pred, P_target, Q_target):
        """
        Compute the loss function.

        Parameters:
        - P_pred: Policy output from the network 
        - Q_pred: Value output from the network 
        - P_target: Target policy from MCTS 
        - Q_target: Target value from MCTS

        Returns:
        - Total loss (weighted sum of policy loss and value loss)
        """
        
        # Make sure P_target is the same shape as P_pred
        P_target = P_target.unsqueeze(0) if P_target.dim() == 1 else P_target  # Ensures (1, 81)

        # Compute policy loss
        policy_loss = self.policy_loss_fn(P_pred, P_target)

        # Compute value loss
        value_loss = self.value_loss_fn(Q_pred.squeeze(), Q_target.squeeze())

        # Total loss (equal weighting)
        total_loss = policy_loss + value_loss

        return total_loss

    def save(self, path="game_network.pth"):
        """ Save model weights """
        torch.save(self.state_dict(), path)

    def load(self, path="game_network.pth"):
        """ Load model weights and set to evaluation mode """
        self.load_state_dict(torch.load(path))
        self.eval()

    
