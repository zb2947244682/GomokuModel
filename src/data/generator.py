"""
五子棋训练数据生成器
支持多种策略生成高质量训练数据
"""

import numpy as np
import logging
import time
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

from ..game.board import GomokuBoard
from ..game.rules import GomokuRules
from ..utils.mcts import SimpleMCTS
from .augmentation import DataAugmentation

class DataGenerator:
    """
    五子棋训练数据生成器
    """
    
    def __init__(self, board_size: int = 15, use_mcts: bool = True, 
                 mcts_simulations: int = 50):
        """
        初始化数据生成器
        
        Args:
            board_size: 棋盘大小
            use_mcts: 是否使用MCTS
            mcts_simulations: MCTS模拟次数
        """
        self.board_size = board_size
        self.use_mcts = use_mcts
        self.mcts_simulations = mcts_simulations
        
        self.rules = GomokuRules(board_size)
        self.augmentation = DataAugmentation()
        self.logger = logging.getLogger(__name__)
        
        if use_mcts:
            self.mcts = SimpleMCTS(
                board_size=board_size,
                num_simulations=mcts_simulations,
                max_time_ms=200  # 减少到200ms以加快生成
            )
        else:
            self.mcts = None
        
    def generate_training_data(self, num_games: int, strategy: str = 'mixed',
                             save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成训练数据
        
        Args:
            num_games: 生成的对局数量
            strategy: 生成策略 ('mcts', 'smart_random', 'mixed')
            save_path: 保存路径
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (状态, 策略, 价值)
        """
        self.logger.info(f"开始生成 {num_games} 局训练数据，策略: {strategy}")
        
        all_states = []
        all_policies = []
        all_values = []
        
        # 根据策略分配对局
        if strategy == 'mixed':
            mcts_games = int(num_games * 0.3)  # 30% MCTS
            smart_games = num_games - mcts_games  # 70% 智能随机
        elif strategy == 'mcts':
            mcts_games = num_games
            smart_games = 0
        elif strategy == 'smart_random':
            mcts_games = 0
            smart_games = num_games
        else:
            raise ValueError(f"未知策略: {strategy}")
        
        # 生成MCTS数据
        if mcts_games > 0 and self.mcts is not None:
            self.logger.info(f"生成 {mcts_games} 局MCTS数据")
            mcts_states, mcts_policies, mcts_values = self._generate_mcts_games(mcts_games)
            all_states.extend(mcts_states)
            all_policies.extend(mcts_policies)
            all_values.extend(mcts_values)
        
        # 生成智能随机数据
        if smart_games > 0:
            self.logger.info(f"生成 {smart_games} 局智能随机数据")
            smart_states, smart_policies, smart_values = self._generate_smart_random_games(smart_games)
            all_states.extend(smart_states)
            all_policies.extend(smart_policies)
            all_values.extend(smart_values)
        
        # 转换为numpy数组
        states = np.array(all_states, dtype=np.float32)
        policies = np.array(all_policies, dtype=np.float32)
        values = np.array(all_values, dtype=np.float32)
        
        # 数据增强
        self.logger.info("应用数据增强...")
        states, policies = self._apply_data_augmentation(states, policies)
        
        # 扩展values以匹配增强后的数据
        values = np.repeat(values, 8)  # 8种变换
        
        self.logger.info(f"数据生成完成: {len(states)} 个样本")
        
        # 保存数据
        if save_path:
            self.save_data(states, policies, values, save_path)
        
        return states, policies, values
    
    def _generate_mcts_games(self, num_games: int) -> Tuple[List, List, List]:
        """
        使用MCTS生成对局数据
        
        Args:
            num_games: 对局数量
            
        Returns:
            Tuple[List, List, List]: (状态列表, 策略列表, 价值列表)
        """
        states = []
        policies = []
        values = []
        
        for game_idx in tqdm(range(num_games), desc="生成MCTS对局"):
            game_states, game_policies, game_result = self._play_mcts_game()
            
            # 计算每个状态的价值
            game_values = self._calculate_state_values(game_states, game_result)
            
            states.extend(game_states)
            policies.extend(game_policies)
            values.extend(game_values)
        
        return states, policies, values
    
    def _generate_smart_random_games(self, num_games: int) -> Tuple[List, List, List]:
        """
        使用智能随机策略生成对局数据
        
        Args:
            num_games: 对局数量
            
        Returns:
            Tuple[List, List, List]: (状态列表, 策略列表, 价值列表)
        """
        states = []
        policies = []
        values = []
        
        for game_idx in tqdm(range(num_games), desc="生成智能随机对局"):
            game_states, game_policies, game_result = self._play_smart_random_game()
            
            # 计算每个状态的价值
            game_values = self._calculate_state_values(game_states, game_result)
            
            states.extend(game_states)
            policies.extend(game_policies)
            values.extend(game_values)
        
        return states, policies, values
    
    def _play_mcts_game(self) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
        """
        使用MCTS进行一局对局
        
        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], int]: (状态列表, 策略列表, 游戏结果)
        """
        board = GomokuBoard(self.board_size)
        states = []
        policies = []
        
        while not board.game_over:
            # 获取当前状态
            state = board.get_state_tensor()
            states.append(state.copy())
            
            # 使用MCTS获取策略
            action_probs = self.mcts.get_action_probabilities(board, temperature=1.0)
            
            # 转换为策略向量
            policy = np.zeros(self.board_size * self.board_size, dtype=np.float32)
            for (row, col), prob in action_probs.items():
                policy[row * self.board_size + col] = prob
            policies.append(policy)
            
            # 选择动作
            if action_probs:
                moves = list(action_probs.keys())
                probs = list(action_probs.values())
                move_idx = np.random.choice(len(moves), p=probs)
                row, col = moves[move_idx]
                board.make_move(row, col)
            else:
                # 如果没有可用动作，随机选择
                legal_moves = board.get_legal_moves()
                if legal_moves:
                    row, col = np.random.choice(len(legal_moves))
                    board.make_move(row, col)
                else:
                    break
        
        return states, policies, board.winner
    
    def _play_smart_random_game(self) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
        """
        使用智能随机策略进行一局对局
        
        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], int]: (状态列表, 策略列表, 游戏结果)
        """
        board = GomokuBoard(self.board_size)
        states = []
        policies = []
        
        while not board.game_over:
            # 获取当前状态
            state = board.get_state_tensor()
            states.append(state.copy())
            
            # 获取智能候选位置
            smart_moves = self.rules.get_smart_moves(board, top_k=10)
            
            if not smart_moves:
                smart_moves = board.get_legal_moves()
            
            if smart_moves:
                # 创建策略分布
                policy = np.zeros(self.board_size * self.board_size, dtype=np.float32)
                
                # 给智能位置分配概率
                total_prob = 1.0
                for i, (row, col) in enumerate(smart_moves):
                    # 前面的位置获得更高概率
                    prob = total_prob * (0.5 ** i)
                    policy[row * self.board_size + col] = prob
                    total_prob -= prob
                    if total_prob <= 0:
                        break
                
                # 归一化
                if policy.sum() > 0:
                    policy = policy / policy.sum()
                
                policies.append(policy)
                
                # 选择动作
                row, col = smart_moves[0]  # 选择最佳位置
                board.make_move(row, col)
            else:
                break
        
        return states, policies, board.winner
    

    
    def _calculate_state_values(self, states: List[np.ndarray], game_result: int) -> List[float]:
        """
        计算状态价值
        
        Args:
            states: 状态列表
            game_result: 游戏结果
            
        Returns:
            List[float]: 价值列表
        """
        values = []
        
        for i, state in enumerate(states):
            # 当前玩家（从状态中推断）
            current_player = 1 if i % 2 == 0 else -1
            
            if game_result == 0:  # 平局
                value = 0.0
            elif game_result == current_player:  # 当前玩家获胜
                # 越早的状态价值越高（因为更接近胜利）
                value = 1.0 - (i * 0.01)  # 轻微衰减
            else:  # 当前玩家失败
                value = -1.0 + (i * 0.01)  # 轻微衰减
            
            # 限制在[-1, 1]范围内
            value = max(-1.0, min(1.0, value))
            values.append(value)
        
        return values
    
    def _apply_data_augmentation(self, states: np.ndarray, 
                               policies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用数据增强
        
        Args:
            states: 状态数组
            policies: 策略数组
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 增强后的(状态, 策略)
        """
        augmented_states = []
        augmented_policies = []
        
        for state, policy in zip(states, policies):
            # 将策略重塑为棋盘形状
            policy_board = policy.reshape(self.board_size, self.board_size)
            
            # 获取所有变换
            transformations = self.augmentation.get_all_transformations()
            
            for transform_func in transformations:
                # 变换状态
                aug_state = transform_func(state)
                
                # 变换策略
                aug_policy_board = transform_func(policy_board)
                aug_policy = aug_policy_board.flatten()
                
                augmented_states.append(aug_state)
                augmented_policies.append(aug_policy)
        
        return np.array(augmented_states), np.array(augmented_policies)
    
    def save_data(self, states: np.ndarray, policies: np.ndarray, 
                  values: np.ndarray, save_path: str):
        """
        保存训练数据
        
        Args:
            states: 状态数组
            policies: 策略数组
            values: 价值数组
            save_path: 保存路径
        """
        try:
            np.savez_compressed(
                save_path,
                states=states,
                policies=policies,
                values=values
            )
            self.logger.info(f"训练数据已保存到: {save_path}")
            
        except Exception as e:
            self.logger.error(f"保存数据失败: {str(e)}")
            raise
    
    def load_data(self, load_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加载训练数据
        
        Args:
            load_path: 加载路径
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (状态, 策略, 价值)
        """
        try:
            data = np.load(load_path)
            states = data['states']
            policies = data['policies']
            values = data['values']
            
            self.logger.info(f"从 {load_path} 加载了 {len(states)} 个样本")
            return states, policies, values
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
            raise
    
    def get_data_statistics(self, states: np.ndarray, policies: np.ndarray, 
                          values: np.ndarray) -> Dict:
        """
        获取数据统计信息
        
        Args:
            states: 状态数组
            policies: 策略数组
            values: 价值数组
            
        Returns:
            Dict: 统计信息
        """
        stats = {
            'num_samples': len(states),
            'state_shape': states.shape,
            'policy_shape': policies.shape,
            'value_shape': values.shape,
            'value_mean': float(np.mean(values)),
            'value_std': float(np.std(values)),
            'value_min': float(np.min(values)),
            'value_max': float(np.max(values)),
            'policy_sparsity': float(np.mean(policies == 0)),
            'memory_usage_mb': (states.nbytes + policies.nbytes + values.nbytes) / (1024 * 1024)
        }
        
        return stats