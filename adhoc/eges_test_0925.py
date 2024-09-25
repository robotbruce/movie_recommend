import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Example data
user_item_behaviors = [(0, 101), (0, 102), (1, 101), (2, 103), (3, 104)]
user_jobcat = {0: 10, 1: 15, 3: 10}  # Not every user has jobcat

# Define the number of users, items, job categories
num_users = 4  # Example user IDs range from 0 to 3
num_jobcats = 20  # Suppose we have 20 job categories

# Map item IDs to indices and update behaviors
item_id_map = {item_id: idx for idx, item_id in enumerate(sorted(set([item for _, item in user_item_behaviors])))}
num_items = len(item_id_map)
user_item_behaviors = [(user_id, item_id_map[item_id]) for user_id, item_id in user_item_behaviors]

print(f"Item ID mapping: {item_id_map}")

# Define embedding dimensions
embedding_dim = 32  # Embedding size for users and items
jobcat_embedding_dim = 16  # Embedding size for job categories

class UserItemDataset(Dataset):
    def __init__(self, user_item_behaviors, user_jobcat):
        self.user_item_behaviors = user_item_behaviors
        self.user_jobcat = user_jobcat

    def __len__(self):
        return len(self.user_item_behaviors)

    def __getitem__(self, idx):
        user_id, item_id = self.user_item_behaviors[idx]
        jobcat_id = self.user_jobcat.get(user_id, -1)  # 使用 -1 表示沒有 jobcat
        return user_id, item_id, jobcat_id

dataset = UserItemDataset(user_item_behaviors, user_jobcat)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

class EGESModel(nn.Module):
    def __init__(self, num_users, num_items, num_jobcats, embedding_dim, jobcat_embedding_dim):
        super(EGESModel, self).__init__()

        # Embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.jobcat_embeddings = nn.Embedding(num_jobcats + 1, jobcat_embedding_dim)  # 添加一個額外的embedding給 -1 的 jobcat

        # Project jobcat embedding to match user embedding dimension
        self.jobcat_projection = nn.Linear(jobcat_embedding_dim, embedding_dim)

        # Attention mechanism
        self.attention_linear = nn.Linear(embedding_dim, 1)

    def forward(self, user_id, item_id, jobcat_id=None):
        # Check if jobcat_id values are within the valid range
        if jobcat_id.max() >= self.jobcat_embeddings.weight.size(0):
            raise ValueError("jobcat_id contains an index out of range for jobcat_embeddings.")
        
        # Print statements to debug the input ranges
        print(f"user_id: {user_id}, max_user_id: {num_users - 1}")
        print(f"item_id: {item_id}, max_item_id: {num_items - 1}")
        print(f"jobcat_id: {jobcat_id}, max_jobcat_id: {num_jobcats - 1}")

        # Ensure all inputs are within valid range
        if (user_id >= num_users).any() or (item_id >= num_items).any():
            raise IndexError("user_id or item_id out of range in self")
        if jobcat_id is not None and ((jobcat_id >= num_jobcats + 1).any() and (jobcat_id != -1).any()):
            raise IndexError("jobcat_id out of range in self")

        user_emb = self.user_embeddings(user_id)
        item_emb = self.item_embeddings(item_id)

        # Check if jobcat_id is provided and valid
        if jobcat_id is not None:
            valid_jobcat_mask = (jobcat_id != -1)  # Create a mask for valid jobcat entries
            jobcat_emb = torch.zeros((jobcat_id.size(0), self.jobcat_embeddings.embedding_dim), device=user_emb.device)
            jobcat_emb[valid_jobcat_mask] = self.jobcat_embeddings(jobcat_id[valid_jobcat_mask])
            jobcat_emb = self.jobcat_projection(jobcat_emb)

            # Attention weight calculation
            attention_weight = torch.zeros(user_emb.size(0), 1, device=user_emb.device)  # Initialize with zeros
            attention_weight[valid_jobcat_mask] = torch.sigmoid(self.attention_linear(jobcat_emb[valid_jobcat_mask]))

            # Combine embeddings
            combined_emb = user_emb + attention_weight * jobcat_emb
        else:
            combined_emb = user_emb

        # Final embedding by combining user and item embeddings
        output = combined_emb * item_emb
        return output.sum(dim=1)


# Initialize the model
model = EGESModel(num_users=num_users,
                  num_items=num_items,
                  num_jobcats=num_jobcats,
                  embedding_dim=embedding_dim,
                  jobcat_embedding_dim=jobcat_embedding_dim)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, dataloader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        for user_id, item_id, jobcat_id in dataloader:
            # Print debug info before the forward pass
            print(f"Training batch - user_id: {user_id}, item_id: {item_id}, jobcat_id: {jobcat_id}")

            # Simulate positive and negative samples
            pos_labels = torch.ones(len(user_id))
            neg_labels = torch.zeros(len(user_id))

            # Forward pass
            pos_output = model(user_id, item_id, jobcat_id)
            neg_output = model(user_id, torch.randint(0, num_items, item_id.shape), jobcat_id)

            # Calculate loss
            loss = criterion(pos_output, pos_labels) + criterion(neg_output, neg_labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

train_model(model, dataloader, criterion, optimizer, epochs=10)
