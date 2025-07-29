"""
五子棋游戏规则实现
"""

import numpy as np
from typing import List, Tuple, Optional
from .board import GomokuBoard

class GomokuRules:
    """
    五子棋游戏规则类，实现自由五子棋规则（无禁手）
    """
    
    def __init__(self, board_size: int = 15):
        """
        初始化游戏规则
        
        Args:
            board_size: 棋盘大小
        """
        self.board_size = board_size
        
    def is_game_over(self, board: GomokuBoard) -> bool:
        """
        判断游戏是否结束
        
        Args:
            board: 棋盘对象
            
        Returns:
            bool: 游戏是否结束
        """
        return board.game_over
    
    def get_winner(self, board: GomokuBoard) -> int:
        """
        获取游戏胜者
        
        Args:
            board: 棋盘对象
            
        Returns:
            int: 胜者 (1=黑胜, -1=白胜, 0=未结束, 2=平局)
        """
        return board.winner
    
    def evaluate_position(self, board: GomokuBoard, player: int) -> float:
        """
        评估当前局面对指定玩家的价值
        
        Args:
            board: 棋盘对象
            player: 玩家 (1=黑, -1=白)
            
        Returns:
            float: 局面评估值，范围[-1, 1]
        """
        if board.game_over:
            if board.winner == player:
                return 1.0
            elif board.winner == -player:
                return -1.0
            else:
                return 0.0  # 平局
                
        # 简单的启发式评估
        score = 0.0
        
        # 评估所有可能的连线
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board.board[row, col] == 0:  # 空位
                    # 评估在此位置落子的价值
                    score += self._evaluate_position_value(board, row, col, player)
                    
        return np.tanh(score / 100.0)  # 归一化到[-1, 1]
    
    def _evaluate_position_value(self, board: GomokuBoard, row: int, col: int, player: int) -> float:
        """
        评估在指定位置落子的价值
        
        Args:
            board: 棋盘对象
            row: 行坐标
            col: 列坐标
            player: 玩家
            
        Returns:
            float: 位置价值
        """
        value = 0.0
        
        # 四个方向
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            # 计算该方向上的连子情况
            my_count, opp_count = self._count_line(board, row, col, dr, dc, player)
            
            # 根据连子数量给分
            if my_count >= 4:
                value += 1000  # 能形成五连
            elif my_count == 3:
                value += 100   # 活三或冲四
            elif my_count == 2:
                value += 10    # 活二或冲三
            elif my_count == 1:
                value += 1     # 单子
                
            # 阻挡对手
            if opp_count >= 4:
                value += 500   # 必须阻挡对手五连
            elif opp_count == 3:
                value += 50    # 阻挡对手活三
            elif opp_count == 2:
                value += 5     # 阻挡对手活二
                
        return value
    
    def _count_line(self, board: GomokuBoard, row: int, col: int, 
                   dr: int, dc: int, player: int) -> Tuple[int, int]:
        """
        计算指定方向上的连子数量
        
        Args:
            board: 棋盘对象
            row: 起始行
            col: 起始列
            dr: 行方向增量
            dc: 列方向增量
            player: 玩家
            
        Returns:
            Tuple[int, int]: (己方连子数, 对方连子数)
        """
        my_count = 0
        opp_count = 0
        
        # 正方向计数
        r, c = row + dr, col + dc
        while (0 <= r < self.board_size and 0 <= c < self.board_size):
            if board.board[r, c] == player:
                my_count += 1
            elif board.board[r, c] == -player:
                opp_count += 1
            else:
                break
            r, c = r + dr, c + dc
            
        # 反方向计数
        r, c = row - dr, col - dc
        while (0 <= r < self.board_size and 0 <= c < self.board_size):
            if board.board[r, c] == player:
                my_count += 1
            elif board.board[r, c] == -player:
                opp_count += 1
            else:
                break
            r, c = r - dr, c - dc
            
        return my_count, opp_count
    
    def get_smart_moves(self, board: GomokuBoard, top_k: int = 10) -> List[Tuple[int, int]]:
        """
        获取智能落子候选位置
        
        Args:
            board: 棋盘对象
            top_k: 返回前k个最佳位置
            
        Returns:
            List[Tuple[int, int]]: 候选位置列表
        """
        if board.game_over:
            return []
            
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return []
            
        # 如果是开局，优先选择中心区域
        if len(board.move_history) < 3:
            center = self.board_size // 2
            center_moves = []
            for row, col in legal_moves:
                distance = abs(row - center) + abs(col - center)
                if distance <= 2:  # 中心5x5区域
                    center_moves.append((row, col))
            if center_moves:
                return center_moves[:top_k]
                
        # 评估所有合法位置
        move_values = []
        for row, col in legal_moves:
            value = self._evaluate_position_value(board, row, col, board.current_player)
            move_values.append(((row, col), value))
            
        # 按价值排序
        move_values.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前top_k个位置
        return [move for move, _ in move_values[:top_k]]
    
    def simulate_random_game(self, board: GomokuBoard) -> int:
        """
        从当前局面开始随机模拟游戏到结束
        
        Args:
            board: 棋盘对象
            
        Returns:
            int: 游戏结果 (1=黑胜, -1=白胜, 0=平局)
        """
        sim_board = board.copy()
        
        # 随机走子直到游戏结束
        max_moves = 50  # 限制模拟深度
        moves_count = 0
        
        while not sim_board.game_over and moves_count < max_moves:
            # 获取智能候选位置
            smart_moves = self.get_smart_moves(sim_board, top_k=5)
            if not smart_moves:
                break
                
            # 从候选位置中随机选择
            move = np.random.choice(len(smart_moves))
            row, col = smart_moves[move]
            
            sim_board.make_move(row, col)
            moves_count += 1
            
        if sim_board.game_over:
            return sim_board.winner if sim_board.winner != 2 else 0
        else:
            # 超时平局
            return 0