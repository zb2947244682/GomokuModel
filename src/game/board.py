"""
五子棋棋盘逻辑实现
"""

import numpy as np
from typing import List, Tuple, Optional

class GomokuBoard:
    """
    五子棋棋盘类，实现基本的棋盘操作和状态管理
    """
    
    def __init__(self, size: int = 15):
        """
        初始化棋盘
        
        Args:
            size: 棋盘大小，默认15x15
        """
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)  # 0=空, 1=黑, -1=白
        self.current_player = 1  # 1=黑子先手, -1=白子
        self.move_history = []  # 落子历史
        self.game_over = False
        self.winner = 0  # 0=未结束, 1=黑胜, -1=白胜, 2=平局
        
    def reset(self):
        """
        重置棋盘到初始状态
        """
        self.board.fill(0)
        self.current_player = 1
        self.move_history.clear()
        self.game_over = False
        self.winner = 0
        
    def is_valid_move(self, row: int, col: int) -> bool:
        """
        检查落子是否合法
        
        Args:
            row: 行坐标
            col: 列坐标
            
        Returns:
            bool: 是否为合法落子
        """
        if self.game_over:
            return False
            
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
            
        return self.board[row, col] == 0
    
    def make_move(self, row: int, col: int) -> bool:
        """
        执行落子操作
        
        Args:
            row: 行坐标
            col: 列坐标
            
        Returns:
            bool: 是否成功落子
        """
        if not self.is_valid_move(row, col):
            return False
            
        # 落子
        self.board[row, col] = self.current_player
        self.move_history.append((row, col, self.current_player))
        
        # 检查游戏是否结束
        if self._check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
        elif len(self.move_history) >= self.size * self.size:
            # 棋盘满了，平局
            self.game_over = True
            self.winner = 2
        else:
            # 切换玩家
            self.current_player = -self.current_player
            
        return True
    
    def _check_win(self, row: int, col: int) -> bool:
        """
        检查指定位置的落子是否形成五连
        
        Args:
            row: 落子行坐标
            col: 落子列坐标
            
        Returns:
            bool: 是否形成五连
        """
        player = self.board[row, col]
        
        # 四个方向：水平、垂直、主对角线、副对角线
        directions = [
            (0, 1),   # 水平
            (1, 0),   # 垂直
            (1, 1),   # 主对角线
            (1, -1)   # 副对角线
        ]
        
        for dr, dc in directions:
            count = 1  # 包含当前落子
            
            # 正方向计数
            r, c = row + dr, col + dc
            while (0 <= r < self.size and 0 <= c < self.size and 
                   self.board[r, c] == player):
                count += 1
                r, c = r + dr, c + dc
                
            # 反方向计数
            r, c = row - dr, col - dc
            while (0 <= r < self.size and 0 <= c < self.size and 
                   self.board[r, c] == player):
                count += 1
                r, c = r - dr, c - dc
                
            if count >= 5:
                return True
                
        return False
    
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """
        获取所有合法落子位置
        
        Returns:
            List[Tuple[int, int]]: 合法位置列表
        """
        if self.game_over:
            return []
            
        legal_moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row, col] == 0:
                    legal_moves.append((row, col))
                    
        return legal_moves
    
    def get_state_tensor(self) -> np.ndarray:
        """
        获取当前棋盘状态的张量表示
        用于神经网络输入：15x15x3
        
        Returns:
            np.ndarray: 形状为(15, 15, 3)的状态张量
        """
        state = np.zeros((self.size, self.size, 3), dtype=np.float32)
        
        # 第一层：当前玩家的棋子
        state[:, :, 0] = (self.board == self.current_player).astype(np.float32)
        
        # 第二层：对手的棋子
        state[:, :, 1] = (self.board == -self.current_player).astype(np.float32)
        
        # 第三层：当前玩家标识（全1或全0）
        state[:, :, 2] = float(self.current_player == 1)
        
        return state
    
    def copy(self) -> 'GomokuBoard':
        """
        创建棋盘的深拷贝
        
        Returns:
            GomokuBoard: 棋盘副本
        """
        new_board = GomokuBoard(self.size)
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.move_history = self.move_history.copy()
        new_board.game_over = self.game_over
        new_board.winner = self.winner
        
        return new_board
    
    def undo_move(self) -> bool:
        """
        撤销上一步落子
        
        Returns:
            bool: 是否成功撤销
        """
        if not self.move_history:
            return False
            
        # 获取最后一步
        row, col, player = self.move_history.pop()
        
        # 撤销落子
        self.board[row, col] = 0
        self.current_player = player
        self.game_over = False
        self.winner = 0
        
        return True
    
    def __str__(self) -> str:
        """
        棋盘的字符串表示
        """
        symbols = {0: '.', 1: '●', -1: '○'}
        lines = []
        
        # 列号标题
        header = '  ' + ''.join(f'{i:2}' for i in range(self.size))
        lines.append(header)
        
        # 棋盘内容
        for i, row in enumerate(self.board):
            line = f'{i:2}' + ''.join(f' {symbols[cell]}' for cell in row)
            lines.append(line)
            
        return '\n'.join(lines)