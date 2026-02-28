from __future__ import annotations  # 启用未来注解特性

from typing import List  # 类型提示
import torch  # PyTorch 张量库
import torch.nn as nn  # 神经网络模块



class CRF(nn.Module):
    """
    Linear-chain CRF with explicit <START>/<STOP> tags included in tag set.

    Args:
        num_tags: number of tags in tag2id (including <START>, <STOP>)
        start_tag_id: id of <START>
        stop_tag_id: id of <STOP>

    Inputs:
        emissions: (B, T, num_tags) - emission scores for each tag at each position
        tags:      (B, T)           - gold tag ids
        mask:      (B, T)           - 1 for valid positions, 0 for invalid/pad/special

    Notes:
        - mask is critical: CRF only considers positions where mask==1.
        - We assume <START>/<STOP> will never appear in tags at valid positions.
    """

    def __init__(self, num_tags: int, start_tag_id: int, stop_tag_id: int):
        super().__init__()  # 调用父类初始化
        if num_tags <= 0:
            raise ValueError("num_tags must be positive")  # 验证标签数量
        self.num_tags = num_tags  # 标签总数
        self.start_tag_id = start_tag_id  # 开始标签 ID
        self.stop_tag_id = stop_tag_id  # 结束标签 ID

        # transitions[i, j] = 从标签 i 转移到标签 j 的分数
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()  # 初始化转移矩阵参数


    def reset_parameters(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)  # 均匀初始化转移分数

        # 可选的硬约束（更安全）：
        # 禁止转移到 START，禁止从 STOP 转移
        self.transitions.data[:, self.start_tag_id] = -1e4  # 禁止转到 START
        self.transitions.data[self.stop_tag_id, :] = -1e4  # 禁止从 STOP 转出


    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        计算负对数似然（用于训练）
        返回: 负对数似然的标量均值
        """
        if emissions.dim() != 3:
            raise ValueError(f"emissions must have shape (B,T,C), got {emissions.shape}")  # 验证发射分数维度
        if tags.dim() != 2:
            raise ValueError(f"tags must have shape (B,T), got {tags.shape}")  # 验证标签维度
        if mask.dim() != 2:
            raise ValueError(f"mask must have shape (B,T), got {mask.shape}")  # 验证掩码维度

        mask = mask.to(dtype=torch.bool)  # 转换为布尔掩码

        log_Z = self._compute_log_partition_function(emissions, mask)   # 计算配分函数对数 (B,)
        gold_score = self._compute_gold_score(emissions, tags, mask)    # 计算正确路径分数 (B,)
        nll = log_Z - gold_score  # 负对数似然 = logZ - gold_score
        return nll.mean()  # 返回批次平均 NLL


    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """
        维特比译码

        返回:
            对于每个批次项，返回最优标签序列（长度 = 有效位置数）。
            仅返回 mask==1 位置的标签。
        """
        mask = mask.to(dtype=torch.bool)  # 转换为布尔掩码
        best_paths = self._viterbi_decode(emissions, mask)  # 执行维特比译码
        return best_paths  # 返回最优路径列表


    def _compute_gold_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """计算正确路径的总分数（用于训练目标）"""
        B, T, C = emissions.shape  # 批量大小、时间步、标签数
        # 每个批次项的分数
        score = emissions.new_zeros(B)  # 初始化为全 0

        # start -> first_tag
        # 假设每个序列至少有 1 个有效位置
        # 初始化上一个标签为 START（对所有批次）
        prev_tag = torch.full((B,), self.start_tag_id, dtype=torch.long, device=emissions.device)

        for t in range(T):
            is_valid = mask[:, t]  # 当前时间步是否有效
            curr_tag = tags[:, t]  # 当前标签

            # (b,t,curr_tag) 位置的发射分数
            emit_score = emissions[:, t].gather(1, curr_tag.unsqueeze(1)).squeeze(1)
            trans_score = self.transitions[prev_tag, curr_tag]  # 转移分数

            # 仅在有效位置添加分数
            score = score + (emit_score + trans_score) * is_valid.to(emissions.dtype)

            # 仅在有效位置更新前一标签
            prev_tag = torch.where(is_valid, curr_tag, prev_tag)

        # last_tag -> STOP (仅对有任何有效令牌的序列)
        score = score + self.transitions[prev_tag, self.stop_tag_id]
        return score  # 返回每个批次项的分数


    def _compute_log_partition_function(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        前向算法计算每个批次项的配分函数对数 logZ。
        返回: (B,)
        """
        B, T, C = emissions.shape  # 批量大小、时间步、标签数

        # alpha: (B, C)，初始化为 -inf
        alpha = emissions.new_full((B, C), -1e4)  # 初始为很小的数
        alpha[:, self.start_tag_id] = 0.0  # START 状态分数为 0

        for t in range(T):
            is_valid = mask[:, t].unsqueeze(1)  # (B,1) 有效标志

            emit_t = emissions[:, t].unsqueeze(1)          # (B,1,C_to) 发射分数
            trans = self.transitions.unsqueeze(0)          # (1,C_from,C_to) 转移矩阵
            alpha_expanded = alpha.unsqueeze(2)            # (B,C_from,1) 扩展前一状态

            # 所有转移的分数: (B, C_from, C_to)
            scores = alpha_expanded + trans + emit_t

            new_alpha = torch.logsumexp(scores, dim=1)     # (B, C_to) 对所有前一状态求 logsumexp

            # 如果位置 t 无效 (mask=0)，保持 alpha 不变
            alpha = torch.where(is_valid, new_alpha, alpha)

        # 转移到 STOP
        alpha = alpha + self.transitions[:, self.stop_tag_id].unsqueeze(0)  # (B,C)
        logZ = torch.logsumexp(alpha, dim=1)  # (B,) 对所有最终状态求 logsumexp
        return logZ  # 返回配分函数对数


    def _viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """维特比算法进行 CRF 译码"""
        B, T, C = emissions.shape  # 批量大小、时间步、标签数

        # 初始化
        score = emissions.new_full((B, C), -1e4)  # 初始化分数为很小的值
        score[:, self.start_tag_id] = 0.0  # START 状态分数为 0

        backpointers: List[torch.Tensor] = []  # 保存回溯指针

        for t in range(T):
            is_valid = mask[:, t].unsqueeze(1)  # (B,1) 有效标志

            emit_t = emissions[:, t].unsqueeze(1)   # (B,1,C_to) 发射分数
            trans = self.transitions.unsqueeze(0)   # (1,C_from,C_to) 转移矩阵

            # (B, C_from, C_to) 所有可能的下一状态分数
            next_score = score.unsqueeze(2) + trans + emit_t

            # 找出每个当前标签的最佳前一标签及其分数
            best_score, best_path = next_score.max(dim=1)  # (B,C_to), (B,C_to)

            # 如果无效位置，保持原分数并存储哑回溯指针
            score = torch.where(is_valid, best_score, score)

            backpointers.append(best_path)  # 保存当前时间步的回溯指针

        # 转移到 STOP
        score = score + self.transitions[:, self.stop_tag_id].unsqueeze(0)  # (B,C)

        # 找出最优最后标签
        best_last_score, best_last_tag = score.max(dim=1)  # (B,)

        # 回溯
        best_paths: List[List[int]] = []  # 最优路径列表
        for b in range(B):
            seq_len = int(mask[b].sum().item())  # 该序列的有效长度
            if seq_len == 0:
                best_paths.append([])  # 空序列
                continue

            last_tag = best_last_tag[b].item()  # 最后标签
            path = []  # 当前路径

            # 从后向前迭代，仅对有效长度
            for t in range(T - 1, -1, -1):
                if not mask[b, t]:  # 如果无效位置，跳过
                    continue
                path.append(last_tag)  # 添加到路径
                last_tag = backpointers[t][b, last_tag].item()  # 回溯到前一标签

            path.reverse()  # 反转路径以得到正向顺序
            # 路径长度应等于 seq_len
            best_paths.append(path)  # 添加到最优路径列表

        return best_paths  # 返回所有最优路径
