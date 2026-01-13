# ec_transformer.py
import torch
import torch.nn as nn
import math
from typing import Optional


# (我们移除了 torch.utils.checkpoint.checkpoint)

class PositionalEncoding(nn.Module):
    """ Standard fixed sinusoidal positional encoding """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ErrorConcealmentTransformer(nn.Module):
    """
    Encoder-Only 架构，用于 HAC++ 锚点属性的错误隐藏。
    """

    def __init__(self,
                 feat_dim: int,
                 n_offsets: int,
                 fh_dim: int,
                 model_dim: int = 128,  # (使用简化后的值)
                 nhead: int = 4,  # (使用简化后的值)
                 num_encoder_layers: int = 3,  # (使用简化后的值)
                 dim_feedforward: int = 512,  # (使用简化后的值)
                 max_neighbors: int = 5,
                 dropout: float = 0.1):
        super().__init__()

        assert model_dim % nhead == 0, "model_dim must be divisible by nhead"

        self.model_dim = model_dim
        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.scaling_dim = 6
        self.offset_dim = 3 * n_offsets
        self.attr_dim = feat_dim + self.scaling_dim + self.offset_dim
        self.fh_dim = fh_dim
        self.max_neighbors = max_neighbors

        # --- 输入嵌入层 ---
        self.target_pos_embed = nn.Linear(3, model_dim)
        self.target_fh_embed = nn.Linear(fh_dim, model_dim)
        self.target_feat_embed = nn.Linear(feat_dim, model_dim)
        self.target_scale_embed = nn.Linear(self.scaling_dim, model_dim)
        self.target_offset_embed = nn.Linear(self.offset_dim, model_dim)
        self.neighbor_pos_embed = nn.Linear(3, model_dim)
        self.neighbor_feat_embed = nn.Linear(self.feat_dim, model_dim)
        self.neighbor_scale_embed = nn.Linear(self.scaling_dim, model_dim)
        self.neighbor_offset_embed = nn.Linear(self.offset_dim, model_dim)

        # --- 特殊学习 Tokens ---
        self.mask_token_feat = nn.Parameter(torch.randn(1, 1, model_dim))
        self.mask_token_scale = nn.Parameter(torch.randn(1, 1, model_dim))
        self.mask_token_offset = nn.Parameter(torch.randn(1, 1, model_dim))
        self.query_token = nn.Parameter(torch.randn(1, 1, model_dim))  # (用于 Encoder-Only 架构)

        # --- 类型嵌入 ---
        self.num_token_types = 10  # 0=Query, 1=TPos, 2=TFh, 3=TFeat, 4=TScale, 5=TOffset, 6=NPos, 7=NFeat, 8=NScale, 9=NOffset
        self.type_embedding = nn.Embedding(self.num_token_types, model_dim)

        # --- 位置编码 ---
        # Encoder 序列长度: 1(Query) + 5(Target) + N_neigh*4(Neighbor)
        self.max_seq_len_enc = 1 + 5 + max_neighbors * 4  # (例如 1 + 5 + 5*4 = 26)
        self.positional_encoding_enc = PositionalEncoding(model_dim, dropout, self.max_seq_len_enc)

        # --- Transformer Encoder 模块 ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        encoder_norm = nn.LayerNorm(model_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers, norm=encoder_norm
        )

        # --- 输出层 ---
        self.output_layer = nn.Linear(model_dim, self.attr_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                target_pos: torch.Tensor,
                target_fh: torch.Tensor,
                target_partial_attrs: torch.Tensor,
                target_missing_mask: torch.Tensor,
                neighbor_pos: torch.Tensor,
                neighbor_attrs: torch.Tensor,
                neighbor_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        B = target_pos.shape[0]
        N_max_neighbors = self.max_neighbors
        device = target_pos.device

        input_embeddings = []
        type_ids = []

        # 1. 添加 Query Token (序列的第一个)
        input_embeddings.append(self.query_token.expand(B, -1, -1))
        type_ids.append(torch.full((B, 1), 0, dtype=torch.long, device=device))  # Type ID 0

        # 2. 目标锚点 (Pos, FH)
        input_embeddings.append(self.target_pos_embed(target_pos).unsqueeze(1))
        type_ids.append(torch.full((B, 1), 1, dtype=torch.long, device=device))
        input_embeddings.append(self.target_fh_embed(target_fh).unsqueeze(1))
        type_ids.append(torch.full((B, 1), 2, dtype=torch.long, device=device))

        # 3. 目标锚点属性 (处理掩码)
        target_feat = target_partial_attrs[:, :self.feat_dim]
        target_scale = target_partial_attrs[:, self.feat_dim: self.feat_dim + self.scaling_dim]
        target_offset = target_partial_attrs[:, self.feat_dim + self.scaling_dim:]

        target_feat_emb = self.target_feat_embed(target_feat).unsqueeze(1)
        target_scale_emb = self.target_scale_embed(target_scale).unsqueeze(1)
        target_offset_emb = self.target_offset_embed(target_offset).unsqueeze(1)

        mask_feat = target_missing_mask[:, 0:1].unsqueeze(-1).bool()
        mask_scale = target_missing_mask[:, 1:2].unsqueeze(-1).bool()
        mask_offset = target_missing_mask[:, 2:3].unsqueeze(-1).bool()

        masked_feat_emb = torch.where(mask_feat, self.mask_token_feat.expand(B, -1, -1), target_feat_emb)
        masked_scale_emb = torch.where(mask_scale, self.mask_token_scale.expand(B, -1, -1), target_scale_emb)
        masked_offset_emb = torch.where(mask_offset, self.mask_token_offset.expand(B, -1, -1), target_offset_emb)

        input_embeddings.extend([masked_feat_emb, masked_scale_emb, masked_offset_emb])
        type_ids.extend([
            torch.full((B, 1), 3, dtype=torch.long, device=device),
            torch.full((B, 1), 4, dtype=torch.long, device=device),
            torch.full((B, 1), 5, dtype=torch.long, device=device)
        ])

        # 4. 邻居锚点信息 (处理 padding)
        current_neighbors = neighbor_pos.shape[1]
        if current_neighbors < N_max_neighbors:
            pad_count = N_max_neighbors - current_neighbors
            pos_padding = torch.zeros(B, pad_count, 3, device=device, dtype=neighbor_pos.dtype)
            attrs_padding = torch.zeros(B, pad_count, self.attr_dim, device=device, dtype=neighbor_attrs.dtype)
            neighbor_pos = torch.cat([neighbor_pos, pos_padding], dim=1)
            neighbor_attrs = torch.cat([neighbor_attrs, attrs_padding], dim=1)
            if neighbor_padding_mask is None:
                neighbor_padding_mask = torch.zeros(B, N_max_neighbors, dtype=torch.bool, device=device)
            current_mask = neighbor_padding_mask[:, :current_neighbors]
            pad_mask = torch.ones(B, pad_count, dtype=torch.bool, device=device)
            neighbor_padding_mask = torch.cat([current_mask, pad_mask], dim=1)
        elif neighbor_padding_mask is None:
            neighbor_padding_mask = torch.zeros(B, N_max_neighbors, dtype=torch.bool, device=device)

        # 5. 嵌入邻居信息
        neighbor_pos_flat = neighbor_pos.view(B * N_max_neighbors, 3)
        neighbor_pos_emb = self.neighbor_pos_embed(neighbor_pos_flat).view(B, N_max_neighbors, self.model_dim)

        neighbor_attrs_flat = neighbor_attrs.view(B * N_max_neighbors, self.attr_dim)
        neighbor_feat_flat = neighbor_attrs_flat[:, :self.feat_dim]
        neighbor_scale_flat = neighbor_attrs_flat[:, self.feat_dim: self.feat_dim + self.scaling_dim]
        neighbor_offset_flat = neighbor_attrs_flat[:, self.feat_dim + self.scaling_dim:]

        neighbor_feat_emb = self.neighbor_feat_embed(neighbor_feat_flat).view(B, N_max_neighbors, self.model_dim)
        neighbor_scale_emb = self.neighbor_scale_embed(neighbor_scale_flat).view(B, N_max_neighbors, self.model_dim)
        neighbor_offset_emb = self.neighbor_offset_embed(neighbor_offset_flat).view(B, N_max_neighbors, self.model_dim)

        for i in range(N_max_neighbors):
            input_embeddings.append(neighbor_pos_emb[:, i:i + 1, :])
            type_ids.append(torch.full((B, 1), 6, dtype=torch.long, device=device))
            input_embeddings.append(neighbor_feat_emb[:, i:i + 1, :])
            type_ids.append(torch.full((B, 1), 7, dtype=torch.long, device=device))
            input_embeddings.append(neighbor_scale_emb[:, i:i + 1, :])
            type_ids.append(torch.full((B, 1), 8, dtype=torch.long, device=device))
            input_embeddings.append(neighbor_offset_emb[:, i:i + 1, :])
            type_ids.append(torch.full((B, 1), 9, dtype=torch.long, device=device))

        # 6. 组合序列
        encoder_input_seq = torch.cat(input_embeddings, dim=1)
        seq_len_enc = encoder_input_seq.shape[1]

        assert seq_len_enc == self.max_seq_len_enc, \
            f"Encoder sequence length ({seq_len_enc}) does not match expected ({self.max_seq_len_enc})"

        type_ids_tensor = torch.cat(type_ids, dim=1)
        encoder_input_seq = encoder_input_seq + self.type_embedding(type_ids_tensor)
        encoder_input_seq = self.positional_encoding_enc(encoder_input_seq)

        # 7. 创建 Encoder Padding Mask
        encoder_padding_mask = torch.zeros(B, seq_len_enc, dtype=torch.bool, device=device)
        for i in range(N_max_neighbors):
            # 目标部分共 1 (Query) + 5 (Target) = 6 个 token
            pos_idx = 6 + i * 4
            feat_idx = 7 + i * 4
            scale_idx = 8 + i * 4
            offset_idx = 9 + i * 4
            is_padding = neighbor_padding_mask[:, i]
            encoder_padding_mask[:, pos_idx] = is_padding
            encoder_padding_mask[:, feat_idx] = is_padding
            encoder_padding_mask[:, scale_idx] = is_padding
            encoder_padding_mask[:, offset_idx] = is_padding

        # 8. Encoder 前向传播
        encoder_output = self.transformer_encoder(
            src=encoder_input_seq,
            src_key_padding_mask=encoder_padding_mask
        )  # [B, seq_len_enc, model_dim]

        # 9. 输出层
        query_output = encoder_output[:, 0, :]  # 提取第一个 token (Query Token)
        predicted_attrs = self.output_layer(query_output)  # [B, attr_dim]

        return predicted_attrs  # [B, attr_dim]