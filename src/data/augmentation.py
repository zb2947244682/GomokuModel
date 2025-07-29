"""
数据增强模块
"""

import numpy as np
from typing import List, Tuple

class DataAugmentation:
    """
    五子棋数据增强类
    通过旋转和翻转生成更多训练数据
    """
    
    def __init__(self):
        """
        初始化数据增强器
        """
        pass
    
    def rotate_90(self, state: np.ndarray, policy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        顺时针旋转90度
        
        Args:
            state: 棋盘状态 (H, W, C)
            policy: 策略矩阵 (H, W)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 旋转后的状态和策略
        """
        # 旋转状态（前两个维度）
        rotated_state = np.rot90(state, k=-1, axes=(0, 1))  # k=-1表示顺时针
        
        # 旋转策略
        rotated_policy = np.rot90(policy, k=-1, axes=(0, 1))
        
        return rotated_state, rotated_policy
    
    def rotate_180(self, state: np.ndarray, policy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        旋转180度
        
        Args:
            state: 棋盘状态 (H, W, C)
            policy: 策略矩阵 (H, W)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 旋转后的状态和策略
        """
        rotated_state = np.rot90(state, k=2, axes=(0, 1))
        rotated_policy = np.rot90(policy, k=2, axes=(0, 1))
        
        return rotated_state, rotated_policy
    
    def rotate_270(self, state: np.ndarray, policy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        顺时针旋转270度（逆时针90度）
        
        Args:
            state: 棋盘状态 (H, W, C)
            policy: 策略矩阵 (H, W)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 旋转后的状态和策略
        """
        rotated_state = np.rot90(state, k=1, axes=(0, 1))  # k=1表示逆时针
        rotated_policy = np.rot90(policy, k=1, axes=(0, 1))
        
        return rotated_state, rotated_policy
    
    def flip_horizontal(self, state: np.ndarray, policy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        水平翻转（左右翻转）
        
        Args:
            state: 棋盘状态 (H, W, C)
            policy: 策略矩阵 (H, W)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 翻转后的状态和策略
        """
        flipped_state = np.fliplr(state)
        flipped_policy = np.fliplr(policy)
        
        return flipped_state, flipped_policy
    
    def flip_vertical(self, state: np.ndarray, policy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        垂直翻转（上下翻转）
        
        Args:
            state: 棋盘状态 (H, W, C)
            policy: 策略矩阵 (H, W)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 翻转后的状态和策略
        """
        flipped_state = np.flipud(state)
        flipped_policy = np.flipud(policy)
        
        return flipped_state, flipped_policy
    
    def flip_diagonal(self, state: np.ndarray, policy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        主对角线翻转（转置）
        
        Args:
            state: 棋盘状态 (H, W, C)
            policy: 策略矩阵 (H, W)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 翻转后的状态和策略
        """
        # 转置前两个维度
        flipped_state = np.transpose(state, (1, 0, 2))
        flipped_policy = np.transpose(policy, (1, 0))
        
        return flipped_state, flipped_policy
    
    def flip_anti_diagonal(self, state: np.ndarray, policy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        副对角线翻转
        
        Args:
            state: 棋盘状态 (H, W, C)
            policy: 策略矩阵 (H, W)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 翻转后的状态和策略
        """
        # 先转置，再旋转180度
        transposed_state = np.transpose(state, (1, 0, 2))
        transposed_policy = np.transpose(policy, (1, 0))
        
        flipped_state = np.rot90(transposed_state, k=2, axes=(0, 1))
        flipped_policy = np.rot90(transposed_policy, k=2, axes=(0, 1))
        
        return flipped_state, flipped_policy
    
    def get_all_transformations(self, state: np.ndarray, 
                              policy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        获取所有8种对称变换
        
        Args:
            state: 原始棋盘状态 (H, W, C)
            policy: 原始策略矩阵 (H, W)
            
        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: 8种变换后的(状态, 策略)对
        """
        transformations = []
        
        # 原始数据
        transformations.append((state.copy(), policy.copy()))
        
        # 旋转变换
        state_90, policy_90 = self.rotate_90(state, policy)
        transformations.append((state_90, policy_90))
        
        state_180, policy_180 = self.rotate_180(state, policy)
        transformations.append((state_180, policy_180))
        
        state_270, policy_270 = self.rotate_270(state, policy)
        transformations.append((state_270, policy_270))
        
        # 翻转变换
        state_h, policy_h = self.flip_horizontal(state, policy)
        transformations.append((state_h, policy_h))
        
        state_v, policy_v = self.flip_vertical(state, policy)
        transformations.append((state_v, policy_v))
        
        state_d, policy_d = self.flip_diagonal(state, policy)
        transformations.append((state_d, policy_d))
        
        state_ad, policy_ad = self.flip_anti_diagonal(state, policy)
        transformations.append((state_ad, policy_ad))
        
        return transformations
    
    def augment_batch(self, states: np.ndarray, policies: np.ndarray, 
                     values: np.ndarray, num_augmentations: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对一批数据进行随机增强
        
        Args:
            states: 状态批次 (N, H, W, C)
            policies: 策略批次 (N, H*W)
            values: 价值批次 (N,)
            num_augmentations: 每个样本的增强数量
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 增强后的数据
        """
        batch_size, height, width, channels = states.shape
        
        augmented_states = []
        augmented_policies = []
        augmented_values = []
        
        for i in range(batch_size):
            state = states[i]
            policy = policies[i].reshape(height, width)
            value = values[i]
            
            # 随机选择变换
            all_transforms = self.get_all_transformations(state, policy)
            
            # 随机选择num_augmentations个变换
            selected_indices = np.random.choice(len(all_transforms), 
                                              size=min(num_augmentations, len(all_transforms)), 
                                              replace=False)
            
            for idx in selected_indices:
                aug_state, aug_policy = all_transforms[idx]
                augmented_states.append(aug_state)
                augmented_policies.append(aug_policy.flatten())
                augmented_values.append(value)
        
        return (np.array(augmented_states), 
                np.array(augmented_policies), 
                np.array(augmented_values))
    
    def verify_transformation(self, original_state: np.ndarray, 
                            original_policy: np.ndarray) -> bool:
        """
        验证变换的正确性
        
        Args:
            original_state: 原始状态
            original_policy: 原始策略
            
        Returns:
            bool: 变换是否正确
        """
        try:
            # 获取所有变换
            transformations = self.get_all_transformations(original_state, original_policy)
            
            # 检查变换数量
            if len(transformations) != 8:
                return False
            
            # 检查每个变换的形状
            for state, policy in transformations:
                if state.shape != original_state.shape:
                    return False
                if policy.shape != original_policy.shape:
                    return False
            
            # 检查概率和是否保持不变
            original_sum = np.sum(original_policy)
            for _, policy in transformations:
                if not np.isclose(np.sum(policy), original_sum, rtol=1e-5):
                    return False
            
            return True
            
        except Exception as e:
            print(f"变换验证失败: {e}")
            return False