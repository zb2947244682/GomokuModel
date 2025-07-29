"""
模型训练器实现
"""

import torch
import torch.optim as optim
import numpy as np
import time
import logging
from typing import Dict, List, Tuple
from tqdm import tqdm

from .network import GomokuNet, GomokuLoss

class ModelTrainer:
    """
    五子棋模型训练器
    """
    
    def __init__(self, board_size: int = 15, num_channels: int = 64,
                 num_residual_blocks: int = 4, learning_rate: float = 0.001,
                 batch_size: int = 64, device: str = 'cpu'):
        """
        初始化训练器
        
        Args:
            board_size: 棋盘大小
            num_channels: 网络通道数
            num_residual_blocks: 残差块数量
            learning_rate: 学习率
            batch_size: 批次大小
            device: 设备类型
        """
        self.board_size = board_size
        self.batch_size = batch_size
        self.device = device
        
        # 创建模型
        self.model = GomokuNet(
            board_size=board_size,
            num_channels=num_channels,
            num_residual_blocks=num_residual_blocks
        ).to(device)
        
        # 创建优化器和损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.criterion = GomokuLoss(value_weight=1.0)
        
        # 训练统计
        self.training_stats = {
            'total_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'epochs': 0
        }
        
        logging.info(f"模型创建完成，参数数量: {self.model.get_model_size():,}")
        logging.info(f"模型大小: {self.model.get_model_size_mb():.1f} MB")
        
    def train(self, training_data: Dict[str, np.ndarray], 
              max_time_minutes: int = 10, save_path: str = None) -> GomokuNet:
        """
        训练模型
        
        Args:
            training_data: 训练数据字典
            max_time_minutes: 最大训练时间（分钟）
            save_path: 模型保存路径
            
        Returns:
            GomokuNet: 训练好的模型
        """
        logging.info(f"开始训练，目标时间: {max_time_minutes} 分钟")
        
        # 准备训练数据
        states = torch.FloatTensor(training_data['states']).to(self.device)
        policies = torch.FloatTensor(training_data['policies']).to(self.device)
        values = torch.FloatTensor(training_data['values']).to(self.device)
        
        # 数据维度转换：(N, H, W, C) -> (N, C, H, W)
        states = states.permute(0, 3, 1, 2)
        
        dataset_size = len(states)
        logging.info(f"训练数据大小: {dataset_size}")
        
        # 训练循环
        start_time = time.time()
        max_time_seconds = max_time_minutes * 60
        epoch = 0
        
        self.model.train()
        
        while True:
            epoch += 1
            epoch_start_time = time.time()
            
            # 随机打乱数据
            indices = torch.randperm(dataset_size)
            
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            num_batches = 0
            
            # 批次训练
            for i in range(0, dataset_size, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                batch_states = states[batch_indices]
                batch_policies = policies[batch_indices]
                batch_values = values[batch_indices]
                
                # 前向传播
                policy_logits, value_pred = self.model(batch_states)
                
                # 计算损失
                loss, policy_loss, value_loss = self.criterion(
                    policy_logits, value_pred, batch_policies, batch_values
                )
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 统计
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1
                
                # 检查时间限制
                elapsed_time = time.time() - start_time
                if elapsed_time >= max_time_seconds:
                    logging.info(f"达到时间限制 ({max_time_minutes} 分钟)，停止训练")
                    break
            
            # 计算平均损失
            avg_loss = total_loss / num_batches
            avg_policy_loss = total_policy_loss / num_batches
            avg_value_loss = total_value_loss / num_batches
            
            # 记录统计信息
            self.training_stats['total_loss'].append(avg_loss)
            self.training_stats['policy_loss'].append(avg_policy_loss)
            self.training_stats['value_loss'].append(avg_value_loss)
            self.training_stats['epochs'] = epoch
            
            # 输出训练进度
            epoch_time = time.time() - epoch_start_time
            elapsed_time = time.time() - start_time
            
            logging.info(
                f"Epoch {epoch}: Loss={avg_loss:.4f} "
                f"(Policy={avg_policy_loss:.4f}, Value={avg_value_loss:.4f}) "
                f"Time={epoch_time:.1f}s, Total={elapsed_time:.1f}s"
            )
            
            # 检查时间限制
            if elapsed_time >= max_time_seconds:
                break
                
        # 保存模型
        if save_path:
            self.save_model(save_path)
            logging.info(f"模型已保存到: {save_path}")
            
        total_time = time.time() - start_time
        logging.info(f"训练完成！总用时: {total_time:.1f} 秒，训练轮数: {epoch}")
        
        return self.model
    
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'model_config': {
                'board_size': self.board_size,
                'num_channels': self.model.num_channels,
                'num_residual_blocks': len(self.model.residual_blocks)
            }
        }, path)
    
    def load_model(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
        
        logging.info(f"模型已从 {path} 加载")
    
    def evaluate_model(self, test_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            test_data: 测试数据
            
        Returns:
            Dict[str, float]: 评估结果
        """
        self.model.eval()
        
        states = torch.FloatTensor(test_data['states']).to(self.device)
        policies = torch.FloatTensor(test_data['policies']).to(self.device)
        values = torch.FloatTensor(test_data['values']).to(self.device)
        
        # 数据维度转换
        states = states.permute(0, 3, 1, 2)
        
        with torch.no_grad():
            policy_logits, value_pred = self.model(states)
            loss, policy_loss, value_loss = self.criterion(
                policy_logits, value_pred, policies, values
            )
            
            # 计算准确率
            policy_probs = torch.exp(policy_logits)
            policy_accuracy = self._calculate_policy_accuracy(policy_probs, policies)
            value_accuracy = self._calculate_value_accuracy(value_pred, values)
        
        return {
            'total_loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'policy_accuracy': policy_accuracy,
            'value_accuracy': value_accuracy
        }
    
    def _calculate_policy_accuracy(self, pred_probs: torch.Tensor, 
                                 target_probs: torch.Tensor) -> float:
        """
        计算策略预测准确率
        
        Args:
            pred_probs: 预测概率
            target_probs: 目标概率
            
        Returns:
            float: 准确率
        """
        pred_moves = torch.argmax(pred_probs, dim=1)
        target_moves = torch.argmax(target_probs, dim=1)
        accuracy = (pred_moves == target_moves).float().mean().item()
        return accuracy
    
    def _calculate_value_accuracy(self, pred_values: torch.Tensor, 
                                target_values: torch.Tensor) -> float:
        """
        计算价值预测准确率（符号一致性）
        
        Args:
            pred_values: 预测价值
            target_values: 目标价值
            
        Returns:
            float: 准确率
        """
        pred_signs = torch.sign(pred_values.squeeze())
        target_signs = torch.sign(target_values)
        accuracy = (pred_signs == target_signs).float().mean().item()
        return accuracy