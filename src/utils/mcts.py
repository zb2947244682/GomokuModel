"""
简化的蒙特卡洛树搜索实现
针对CPU性能优化的MCTS算法
"""

import numpy as np
import math
import time
from typing import Dict, List, Tuple, Optional

from ..game.board import GomokuBoard
from ..game.rules import GomokuRules

class MCTSNode:
    """
    MCTS树节点
    """
    
    def __init__(self, board: GomokuBoard, parent: Optional['MCTSNode'] = None, 
                 move: Optional[Tuple[int, int]] = None):
        """
        初始化MCTS节点
        
        Args:
            board: 棋盘状态
            parent: 父节点
            move: 导致此状态的落子
        """
        self.board = board.copy()
        self.parent = parent
        self.move = move
        
        self.children: Dict[Tuple[int, int], 'MCTSNode'] = {}
        self.visits = 0
        self.value_sum = 0.0
        self.is_expanded = False
        
    def is_leaf(self) -> bool:
        """
        判断是否为叶子节点
        """
        return not self.is_expanded
    
    def is_terminal(self) -> bool:
        """
        判断是否为终端节点
        """
        return self.board.game_over
    
    def get_value(self) -> float:
        """
        获取节点平均价值
        """
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    def get_ucb_score(self, exploration_weight: float = 1.4) -> float:
        """
        计算UCB分数
        
        Args:
            exploration_weight: 探索权重
            
        Returns:
            float: UCB分数
        """
        if self.visits == 0:
            return float('inf')
            
        if self.parent is None:
            return self.get_value()
            
        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        
        return self.get_value() + exploration
    
    def select_child(self, exploration_weight: float = 1.4) -> 'MCTSNode':
        """
        选择最佳子节点
        
        Args:
            exploration_weight: 探索权重
            
        Returns:
            MCTSNode: 最佳子节点
        """
        best_score = float('-inf')
        best_child = None
        
        for child in self.children.values():
            score = child.get_ucb_score(exploration_weight)
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child
    
    def expand(self, rules: GomokuRules) -> List['MCTSNode']:
        """
        扩展节点
        
        Args:
            rules: 游戏规则
            
        Returns:
            List[MCTSNode]: 新创建的子节点列表
        """
        if self.is_expanded or self.is_terminal():
            return []
            
        # 获取智能候选位置
        candidate_moves = rules.get_smart_moves(self.board, top_k=8)
        
        if not candidate_moves:
            candidate_moves = self.board.get_legal_moves()
            
        new_children = []
        
        for move in candidate_moves:
            row, col = move
            
            # 创建新的棋盘状态
            new_board = self.board.copy()
            if new_board.make_move(row, col):
                child = MCTSNode(new_board, parent=self, move=move)
                self.children[move] = child
                new_children.append(child)
        
        self.is_expanded = True
        return new_children
    
    def backup(self, value: float):
        """
        回传价值
        
        Args:
            value: 要回传的价值
        """
        self.visits += 1
        self.value_sum += value
        
        if self.parent is not None:
            # 从对手角度看，价值取反
            self.parent.backup(-value)

class SimpleMCTS:
    """
    简化的MCTS实现
    针对CPU性能和训练速度优化
    """
    
    def __init__(self, board_size: int = 15, num_simulations: int = 50,
                 exploration_weight: float = 1.4, max_time_ms: int = 1000):
        """
        初始化MCTS
        
        Args:
            board_size: 棋盘大小
            num_simulations: 模拟次数
            exploration_weight: 探索权重
            max_time_ms: 最大思考时间（毫秒）
        """
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.max_time_ms = max_time_ms
        
        self.rules = GomokuRules(board_size)
        
    def get_action_probabilities(self, board: GomokuBoard, 
                               temperature: float = 1.0) -> Dict[Tuple[int, int], float]:
        """
        获取动作概率分布
        
        Args:
            board: 当前棋盘状态
            temperature: 温度参数，控制探索程度
            
        Returns:
            Dict[Tuple[int, int], float]: 动作概率字典
        """
        if board.game_over:
            return {}
            
        root = MCTSNode(board)
        
        # 执行MCTS搜索
        start_time = time.time() * 1000  # 转换为毫秒
        
        for simulation in range(self.num_simulations):
            # 检查时间限制
            current_time = time.time() * 1000
            if current_time - start_time > self.max_time_ms:
                break
                
            self._simulate(root)
        
        # 计算访问次数分布
        visit_counts = {}
        total_visits = 0
        
        for move, child in root.children.items():
            visit_counts[move] = child.visits
            total_visits += child.visits
        
        if total_visits == 0:
            # 如果没有访问记录，返回均匀分布
            legal_moves = board.get_legal_moves()
            if legal_moves:
                prob = 1.0 / len(legal_moves)
                return {move: prob for move in legal_moves}
            else:
                return {}
        
        # 应用温度参数
        if temperature == 0:
            # 贪心选择
            best_move = max(visit_counts.keys(), key=lambda m: visit_counts[m])
            return {best_move: 1.0}
        else:
            # 根据访问次数计算概率
            probabilities = {}
            
            if temperature == 1.0:
                # 直接按访问次数比例
                for move, visits in visit_counts.items():
                    probabilities[move] = visits / total_visits
            else:
                # 应用温度缩放
                scaled_visits = {}
                for move, visits in visit_counts.items():
                    scaled_visits[move] = visits ** (1.0 / temperature)
                
                total_scaled = sum(scaled_visits.values())
                for move, scaled in scaled_visits.items():
                    probabilities[move] = scaled / total_scaled
            
            return probabilities
    
    def _simulate(self, root: MCTSNode):
        """
        执行一次MCTS模拟
        
        Args:
            root: 根节点
        """
        # 1. 选择阶段
        node = self._select(root)
        
        # 2. 扩展阶段
        if not node.is_terminal() and node.visits > 0:
            children = node.expand(self.rules)
            if children:
                node = np.random.choice(children)
        
        # 3. 模拟阶段
        value = self._rollout(node)
        
        # 4. 回传阶段
        node.backup(value)
    
    def _select(self, root: MCTSNode) -> MCTSNode:
        """
        选择阶段：从根节点向下选择到叶子节点
        
        Args:
            root: 根节点
            
        Returns:
            MCTSNode: 选中的叶子节点
        """
        node = root
        
        while not node.is_leaf() and not node.is_terminal():
            node = node.select_child(self.exploration_weight)
            
        return node
    
    def _rollout(self, node: MCTSNode) -> float:
        """
        模拟阶段：从当前节点随机模拟到游戏结束
        
        Args:
            node: 当前节点
            
        Returns:
            float: 模拟结果价值
        """
        if node.is_terminal():
            # 如果已经是终端状态，直接返回结果
            return self.rules.evaluate_position(node.board, node.board.current_player)
        
        # 快速随机模拟
        result = self.rules.simulate_random_game(node.board)
        
        # 转换为当前玩家的价值
        if result == node.board.current_player:
            return 1.0
        elif result == -node.board.current_player:
            return -1.0
        else:
            return 0.0  # 平局
    
    def get_best_move(self, board: GomokuBoard) -> Optional[Tuple[int, int]]:
        """
        获取最佳落子位置
        
        Args:
            board: 当前棋盘状态
            
        Returns:
            Optional[Tuple[int, int]]: 最佳落子位置
        """
        action_probs = self.get_action_probabilities(board, temperature=0)
        
        if not action_probs:
            return None
            
        return max(action_probs.keys(), key=lambda m: action_probs[m])
    
    def get_move_values(self, board: GomokuBoard) -> Dict[Tuple[int, int], float]:
        """
        获取所有可能落子的价值评估
        
        Args:
            board: 当前棋盘状态
            
        Returns:
            Dict[Tuple[int, int], float]: 落子位置到价值的映射
        """
        root = MCTSNode(board)
        
        # 执行搜索
        for _ in range(self.num_simulations):
            self._simulate(root)
        
        # 返回子节点的价值
        move_values = {}
        for move, child in root.children.items():
            move_values[move] = child.get_value()
            
        return move_values