import torch
import torch.nn as nn


class Selector(nn.Module):
    def __init__(self, item_dim, policy_dim, history_num=50, hidden_dim=64):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(history_num * item_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, policy_dim),
            nn.ReLU()
        )

    def forward(self, user_history, policy_results):
        # user_history: [B, H, item_dim]
        # policy_results: [B, N, policy_dim]

        # Flatten the user_history tensor
        user_history = user_history.view(user_history.shape[0], -1)

        # Compute the dot product between user_history and policy_results
        a = self.fc1(user_history).unsqueeze(2)
        logits = torch.matmul(policy_results, a).squeeze(2)

        # Apply softmax to the logits to get probabilities
        probabilities = torch.softmax(logits, dim=1)

        return probabilities
