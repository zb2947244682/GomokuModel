"""
五子棋神经网络模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class ResidualBlock(nn.Module):
    """
    残差块，用于构建深层网络
    """
    
    def __init__(self, channels: int):
        """
        初始化残差块
        
        Args:
            channels: 通道数
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 输出张量
        """
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual  # 残差连接
        out = F.relu(out)
        
        return out

class GomokuNet(nn.Module):
    """
    五子棋神经网络模型
    包含策略头和价值头，适用于AlphaZero风格的训练
    """
    
    def __init__(self, board_size: int = 15, num_channels: int = 64, 
                 num_residual_blocks: int = 4):
        """
        初始化网络
        
        Args:
            board_size: 棋盘大小
            num_channels: 卷积层通道数
            num_residual_blocks: 残差块数量
        """
        super(GomokuNet, self).__init__()
        
        self.board_size = board_size
        self.num_channels = num_channels
        
        # 输入卷积层
        self.input_conv = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_channels)
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_residual_blocks)
        ])
        
        # 策略头 (Policy Head)
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # 价值头 (Value Head)
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, num_channels)
        self.value_fc2 = nn.Linear(num_channels, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, 3, board_size, board_size)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (策略概率, 价值评估)
        """
        # 输入层
        out = self.input_conv(x)
        out = self.input_bn(out)
        out = F.relu(out)
        
        # 残差块
        for block in self.residual_blocks:
            out = block(out)
            
        # 策略头
        policy = self.policy_conv(out)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(policy.size(0), -1)  # 展平
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)  # 对数概率
        
        # 价值头
        value = self.value_conv(out)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(value.size(0), -1)  # 展平
        value = self.value_fc1(value)
        value = F.relu(value)
        value = self.value_fc2(value)
        value = torch.tanh(value)  # 输出范围[-1, 1]
        
        return policy, value
    
    def predict(self, board_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        对单个棋盘状态进行预测
        
        Args:
            board_state: 棋盘状态，形状为 (board_size, board_size, 3)
            
        Returns:
            Tuple[np.ndarray, float]: (策略概率, 价值评估)
        """
        self.eval()
        
        with torch.no_grad():
            # 转换为张量并添加批次维度
            x = torch.FloatTensor(board_state).permute(2, 0, 1).unsqueeze(0)
            
            # 前向传播
            policy_logits, value = self.forward(x)
            
            # 转换为概率
            policy_probs = torch.exp(policy_logits).squeeze(0).numpy()
            value_scalar = value.squeeze(0).item()
            
        return policy_probs, value_scalar
    
    def get_model_size(self) -> int:
        """
        获取模型参数数量
        
        Returns:
            int: 参数总数
        """
        return sum(p.numel() for p in self.parameters())
    
    def get_model_size_mb(self) -> float:
        """
        获取模型大小（MB）
        
        Returns:
            float: 模型大小（MB）
        """
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        total_size = param_size + buffer_size
        return total_size / (1024 * 1024)  # 转换为MB

class GomokuLoss(nn.Module):
    """
    五子棋模型的损失函数
    结合策略损失和价值损失
    """
    
    def __init__(self, value_weight: float = 1.0):
        """
        初始化损失函数
        
        Args:
            value_weight: 价值损失的权重
        """
        super(GomokuLoss, self).__init__()
        self.value_weight = value_weight
        
    def forward(self, policy_logits: torch.Tensor, value_pred: torch.Tensor,
                policy_target: torch.Tensor, value_target: torch.Tensor) -> torch.Tensor:
        """
        计算总损失
        
        Args:
            policy_logits: 策略预测（对数概率）
            value_pred: 价值预测
            policy_target: 策略目标
            value_target: 价值目标
            
        Returns:
            torch.Tensor: 总损失
        """
        # 策略损失（交叉熵）
        policy_loss = -torch.sum(policy_target * policy_logits, dim=1).mean()
        
        # 价值损失（均方误差）
        value_loss = F.mse_loss(value_pred.squeeze(), value_target)
        
        # 总损失
        total_loss = policy_loss + self.value_weight * value_loss
        
        return total_loss, policy_loss, value_loss