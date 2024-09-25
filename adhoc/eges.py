import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 示例节点嵌入 (通过Node2Vec生成)
node_embeddings = torch.rand(10, 32)  # 10个节点，每个节点32维嵌入

# 示例侧信息嵌入 (假设每个节点有3个侧信息，每个侧信息是16维)
side_info_embeddings = torch.rand(10, 3, 16)  # 10个节点，每个节点有3个16维的侧信息嵌入


class EGES(nn.Module):
    def __init__(self, node_embedding_dim, side_info_dim, attention_dim):
        super(EGES, self).__init__()
        self.node_embedding_dim = node_embedding_dim
        self.side_info_dim = side_info_dim
        self.attention_dim = attention_dim

        # 用于计算注意力权重的线性层
        self.attention_fc = nn.Linear(side_info_dim, attention_dim)
        self.attention_query = nn.Parameter(torch.randn(attention_dim))

        # 最终输出的节点嵌入维度
        self.output_fc = nn.Linear(node_embedding_dim + side_info_dim, node_embedding_dim)

    def forward(self, node_embeddings, side_info_embeddings):
        # side_info_embeddings: (batch_size, num_side_info, side_info_dim)

        # 计算注意力权重
        att_weights = torch.tanh(self.attention_fc(side_info_embeddings))  # (batch_size, num_side_info, attention_dim)
        att_weights = torch.matmul(att_weights, self.attention_query)  # (batch_size, num_side_info)
        att_weights = F.softmax(att_weights, dim=1)  # (batch_size, num_side_info)

        # 计算加权侧信息嵌入
        side_info_weighted = torch.sum(att_weights.unsqueeze(-1) * side_info_embeddings,
                                       dim=1)  # (batch_size, side_info_dim)

        # 将节点嵌入与加权侧信息嵌入拼接
        combined = torch.cat([node_embeddings, side_info_weighted],
                             dim=-1)  # (batch_size, node_embedding_dim + side_info_dim)

        # 通过全连接层生成最终嵌入
        output_embeddings = self.output_fc(combined)  # (batch_size, node_embedding_dim)

        return output_embeddings


# 定义模型
model = EGES(node_embedding_dim=32, side_info_dim=16, attention_dim=8)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 示例目标节点对 (正样本和负样本)
positive_pairs = torch.tensor([[0, 1], [2, 3], [4, 5]])  # 假设3个正样本对
negative_pairs = torch.tensor([[0, 6], [2, 7], [4, 8]])  # 假设3个负样本对

# 正样本标签为1，负样本标签为0
labels = torch.cat([torch.ones(len(positive_pairs)), torch.zeros(len(negative_pairs))])

# 训练示例
for epoch in range(100):  # 训练100个epoch
    optimizer.zero_grad()

    # 获取当前批次的节点嵌入和侧信息嵌入
    batch_node_embeddings = node_embeddings[torch.cat([positive_pairs[:, 0], negative_pairs[:, 0]])]
    batch_side_info_embeddings = side_info_embeddings[torch.cat([positive_pairs[:, 0], negative_pairs[:, 0]])]

    # 生成最终的节点嵌入
    final_embeddings = model(batch_node_embeddings, batch_side_info_embeddings)

    # 计算正样本和负样本的相似度
    pos_scores = (final_embeddings[:len(positive_pairs)] * node_embeddings[positive_pairs[:, 1]]).sum(dim=1)
    neg_scores = (final_embeddings[len(positive_pairs):] * node_embeddings[negative_pairs[:, 1]]).sum(dim=1)

    # 拼接正负样本分数
    scores = torch.cat([pos_scores, neg_scores])

    # 计算损失并反向传播
    loss = criterion(scores, labels)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')




a = torch.tensor([[[1, 2, 3], [3, 4, 5]],[[5, 6, 7], [7, 8, 9]]])
print(a.shape)
print(a)

b = a.reshape(2, 6)
c = a.view(2, 6)
d = a.reshape(2, -1)
print(b.shape, c.shape, d.shape)