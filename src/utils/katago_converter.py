"""
KataGO模型格式转换工具
将PyTorch模型转换为KataGO兼容的bin.gz格式
"""

import torch
import numpy as np
import gzip
import struct
import os
import logging
from typing import Dict, Any, Tuple

from ..model.network import GomokuNet

class KataGoConverter:
    """
    KataGO格式转换器
    """
    
    def __init__(self):
        """
        初始化转换器
        """
        self.logger = logging.getLogger(__name__)
        
    def convert_model(self, model: GomokuNet, output_path: str, 
                     model_name: str = "gomoku_model") -> bool:
        """
        将PyTorch模型转换为KataGO兼容格式
        
        Args:
            model: 训练好的PyTorch模型
            output_path: 输出文件路径
            model_name: 模型名称
            
        Returns:
            bool: 转换是否成功
        """
        try:
            self.logger.info(f"开始转换模型到KataGO格式: {output_path}")
            
            # 确保模型处于评估模式
            model.eval()
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 收集模型权重
            model_data = self._extract_model_weights(model)
            
            # 创建KataGO兼容的模型描述
            model_config = self._create_model_config(model, model_name)
            
            # 写入bin.gz文件
            self._write_katago_format(model_data, model_config, output_path)
            
            # 验证文件大小
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            self.logger.info(f"模型转换完成，文件大小: {file_size:.2f} MB")
            
            if file_size > 120:  # 稍微超过100MB的警告
                self.logger.warning(f"模型文件较大 ({file_size:.2f} MB)，可能影响加载速度")
            
            return True
            
        except Exception as e:
            self.logger.error(f"模型转换失败: {str(e)}")
            return False
    
    def _extract_model_weights(self, model: GomokuNet) -> Dict[str, np.ndarray]:
        """
        提取模型权重
        
        Args:
            model: PyTorch模型
            
        Returns:
            Dict[str, np.ndarray]: 权重字典
        """
        weights = {}
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                # 转换为numpy数组
                weight_array = param.cpu().numpy().astype(np.float32)
                weights[name] = weight_array
                
                self.logger.debug(f"提取权重: {name}, 形状: {weight_array.shape}")
        
        return weights
    
    def _create_model_config(self, model: GomokuNet, model_name: str) -> Dict[str, Any]:
        """
        创建模型配置信息
        
        Args:
            model: PyTorch模型
            model_name: 模型名称
            
        Returns:
            Dict[str, Any]: 模型配置
        """
        config = {
            "model_name": model_name,
            "version": "1.16.3",
            "board_size": model.board_size,
            "input_channels": 3,  # 15x15x3输入
            "policy_output_size": model.board_size * model.board_size,  # 225
            "value_output_size": 1,
            "num_residual_blocks": len([m for m in model.backbone if hasattr(m, 'conv1')]),
            "channels": model.channels,
            "architecture": "residual_cnn",
            "training_info": {
                "framework": "pytorch",
                "optimizer": "adam",
                "data_augmentation": True,
                "game_type": "gomoku_freestyle"
            }
        }
        
        return config
    
    def _write_katago_format(self, weights: Dict[str, np.ndarray], 
                           config: Dict[str, Any], output_path: str):
        """
        写入KataGO格式文件
        
        Args:
            weights: 模型权重
            config: 模型配置
            output_path: 输出路径
        """
        with gzip.open(output_path, 'wb') as f:
            # 写入文件头
            self._write_header(f, config)
            
            # 写入权重数据
            self._write_weights(f, weights)
            
            # 写入配置信息
            self._write_config(f, config)
    
    def _write_header(self, f, config: Dict[str, Any]):
        """
        写入文件头
        
        Args:
            f: 文件对象
            config: 配置信息
        """
        # KataGO魔数
        magic = b'KATAGO_MODEL'
        f.write(magic)
        
        # 版本号
        version = struct.pack('<I', 1)  # 版本1
        f.write(version)
        
        # 棋盘大小
        board_size = struct.pack('<I', config['board_size'])
        f.write(board_size)
        
        # 输入通道数
        input_channels = struct.pack('<I', config['input_channels'])
        f.write(input_channels)
        
        # 策略输出大小
        policy_size = struct.pack('<I', config['policy_output_size'])
        f.write(policy_size)
        
        # 价值输出大小
        value_size = struct.pack('<I', config['value_output_size'])
        f.write(value_size)
    
    def _write_weights(self, f, weights: Dict[str, np.ndarray]):
        """
        写入权重数据
        
        Args:
            f: 文件对象
            weights: 权重字典
        """
        # 权重数量
        num_weights = struct.pack('<I', len(weights))
        f.write(num_weights)
        
        for name, weight in weights.items():
            # 权重名称长度和名称
            name_bytes = name.encode('utf-8')
            name_len = struct.pack('<I', len(name_bytes))
            f.write(name_len)
            f.write(name_bytes)
            
            # 权重形状
            shape = weight.shape
            shape_len = struct.pack('<I', len(shape))
            f.write(shape_len)
            for dim in shape:
                f.write(struct.pack('<I', dim))
            
            # 权重数据
            weight_bytes = weight.tobytes()
            weight_len = struct.pack('<I', len(weight_bytes))
            f.write(weight_len)
            f.write(weight_bytes)
    
    def _write_config(self, f, config: Dict[str, Any]):
        """
        写入配置信息
        
        Args:
            f: 文件对象
            config: 配置字典
        """
        import json
        
        # 将配置转换为JSON字符串
        config_json = json.dumps(config, indent=2)
        config_bytes = config_json.encode('utf-8')
        
        # 写入配置长度和数据
        config_len = struct.pack('<I', len(config_bytes))
        f.write(config_len)
        f.write(config_bytes)
    
    def create_katago_config(self, model_path: str, output_dir: str) -> str:
        """
        创建KataGO配置文件
        
        Args:
            model_path: 模型文件路径
            output_dir: 输出目录
            
        Returns:
            str: 配置文件路径
        """
        config_path = os.path.join(output_dir, "katago_config.cfg")
        
        config_content = f"""# KataGO配置文件 - 五子棋模型
# 生成时间: {self._get_timestamp()}

# 模型文件路径
modelFile = {os.path.abspath(model_path)}

# 游戏规则设置
rules = freestyle  # 自由五子棋规则

# 棋盘设置
boardSizeX = 15
boardSizeY = 15

# 搜索设置
maxVisits = 800
maxPlayouts = 800
maxTime = 10.0

# GPU设置（如果有GPU可用）
numSearchThreads = 1
nnMaxBatchSize = 8
nnCacheSizePowerOfTwo = 16

# 输出设置
logSearchInfo = false
logMoves = true
logToStderr = false

# 分析设置
analysisPVLen = 15
reportAnalysisWinratesAs = SIDETOMOVE

# 其他设置
allowResignation = true
resignThreshold = -0.90
resignConsecTurns = 3

# 随机性设置
searchRandomSeed = 
nnRandomSeed = 

# 时间控制
lagBuffer = 1.0
"""
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_content)
            
            self.logger.info(f"KataGO配置文件已创建: {config_path}")
            return config_path
            
        except Exception as e:
            self.logger.error(f"创建配置文件失败: {str(e)}")
            return ""
    
    def create_integration_guide(self, output_dir: str) -> str:
        """
        创建KataGO集成指南
        
        Args:
            output_dir: 输出目录
            
        Returns:
            str: 指南文件路径
        """
        guide_path = os.path.join(output_dir, "katago_integration_guide.md")
        
        guide_content = f"""# KataGO 五子棋模型集成指南

## 概述
本指南说明如何将训练好的五子棋模型集成到KataGO 1.16.3中使用。

## 文件说明
- `gomoku_model.bin.gz`: 训练好的模型权重文件
- `katago_config.cfg`: KataGO配置文件
- `katago_integration_guide.md`: 本集成指南

## 系统要求
- KataGO 1.16.3或更高版本
- 支持五子棋的KataGO构建版本
- 至少4GB可用内存

## 安装步骤

### 1. 下载KataGO
```bash
# 从官方GitHub下载KataGO 1.16.3
# https://github.com/lightvector/KataGo/releases/tag/v1.16.3
```

### 2. 配置模型
1. 将`gomoku_model.bin.gz`复制到KataGO目录
2. 将`katago_config.cfg`复制到KataGO目录
3. 根据需要修改配置文件中的路径

### 3. 测试模型
```bash
# 测试模型加载
./katago benchmark -config katago_config.cfg

# 运行分析模式
./katago analysis -config katago_config.cfg
```

## 使用方法

### 命令行分析
```bash
# 分析特定局面
./katago analysis -config katago_config.cfg -analysis-threads 1

# 输入SGF格式的棋谱进行分析
echo "(;FF[4]GM[1]SZ[15];B[hh];W[hi])" | ./katago analysis -config katago_config.cfg
```

### GTP协议
```bash
# 启动GTP模式
./katago gtp -config katago_config.cfg

# 在GTP模式下的基本命令:
# boardsize 15
# clear_board
# play black h8
# genmove white
```

## 配置调优

### 性能优化
- 调整`maxVisits`控制搜索深度（推荐800-1600）
- 调整`maxTime`控制思考时间（推荐5-30秒）
- 根据硬件调整`numSearchThreads`

### 游戏规则
- `rules = freestyle`: 自由五子棋（无禁手）
- 棋盘大小固定为15x15

### 内存使用
- `nnCacheSizePowerOfTwo`: 神经网络缓存大小（推荐16-20）
- `nnMaxBatchSize`: 批处理大小（CPU推荐1-8）

## 故障排除

### 常见问题
1. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确认模型文件完整性
   - 检查KataGO版本兼容性

2. **内存不足**
   - 减少`nnCacheSizePowerOfTwo`值
   - 减少`maxVisits`值
   - 关闭其他占用内存的程序

3. **性能问题**
   - 在CPU上运行时，减少搜索线程数
   - 调整批处理大小
   - 考虑使用更小的模型

### 日志分析
- 检查KataGO输出日志中的错误信息
- 启用`logSearchInfo = true`获取详细搜索信息
- 使用`logToStderr = true`在控制台显示日志

## 模型信息
- **架构**: 残差卷积神经网络
- **输入**: 15x15x3 (棋盘状态)
- **输出**: 策略头(225维) + 价值头(1维)
- **训练数据**: 模拟对局数据 + 数据增强
- **优化器**: Adam
- **文件大小**: ~100MB

## 技术支持
如果遇到问题，请检查：
1. KataGO版本是否为1.16.3或更高
2. 模型文件是否完整
3. 配置文件路径是否正确
4. 系统资源是否充足

## 更新日志
- 初始版本: 支持KataGO 1.16.3
- 游戏类型: 自由五子棋
- 生成时间: {self._get_timestamp()}
"""
        
        try:
            with open(guide_path, 'w', encoding='utf-8') as f:
                f.write(guide_content)
            
            self.logger.info(f"集成指南已创建: {guide_path}")
            return guide_path
            
        except Exception as e:
            self.logger.error(f"创建集成指南失败: {str(e)}")
            return ""
    
    def _get_timestamp(self) -> str:
        """
        获取当前时间戳
        
        Returns:
            str: 格式化的时间戳
        """
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def validate_model_size(self, model_path: str, max_size_mb: float = 100.0) -> Tuple[bool, float]:
        """
        验证模型文件大小
        
        Args:
            model_path: 模型文件路径
            max_size_mb: 最大允许大小（MB）
            
        Returns:
            Tuple[bool, float]: (是否符合要求, 实际大小MB)
        """
        try:
            if not os.path.exists(model_path):
                return False, 0.0
            
            size_bytes = os.path.getsize(model_path)
            size_mb = size_bytes / (1024 * 1024)
            
            is_valid = size_mb <= max_size_mb
            
            if is_valid:
                self.logger.info(f"模型大小验证通过: {size_mb:.2f} MB")
            else:
                self.logger.warning(f"模型过大: {size_mb:.2f} MB (限制: {max_size_mb} MB)")
            
            return is_valid, size_mb
            
        except Exception as e:
            self.logger.error(f"验证模型大小失败: {str(e)}")
            return False, 0.0