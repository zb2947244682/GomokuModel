"""五子棋预训练模型主程序
用于训练五子棋神经网络模型并导出为KataGO兼容格式
"""

import os
import sys
import logging
import time
from datetime import datetime

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.generator import DataGenerator
from src.model.trainer import ModelTrainer
from src.utils.katago_converter import KataGoConverter

def setup_logging():
    """
    设置日志配置
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def create_directories():
    """
    创建必要的目录
    """
    directories = [
        'data',
        'models',
        'output',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")

def main():
    """
    主函数
    """
    print("=" * 60)
    print("五子棋预训练模型训练程序")
    print("目标: 生成KataGO 1.16.3兼容的模型文件")
    print("=" * 60)
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 创建目录
    create_directories()
    
    try:
        # 训练参数
        config = {
            'board_size': 15,
            'num_games': 10000,  # 用户要求的训练局数
            'max_training_time': 600,  # 10分钟训练时间
            'batch_size': 32,
            'learning_rate': 0.001,
            'model_channels': 64,  # 控制模型大小
            'num_residual_blocks': 4,  # 4层残差块
            'target_model_size_mb': 100,  # 目标模型大小
            'mcts_simulations': 50,  # MCTS模拟次数（CPU优化）
            'use_mcts_ratio': 0.3  # 30%使用MCTS，70%使用智能随机
        }
        
        logger.info("开始训练流程...")
        logger.info(f"配置参数: {config}")
        
        # 1. 生成训练数据
        logger.info("步骤1: 生成训练数据")
        data_generator = DataGenerator(
            board_size=config['board_size'],
            use_mcts=True,  # 使用MCTS生成高质量数据
            mcts_simulations=config['mcts_simulations']
        )
        
        # 生成数据（混合MCTS和智能随机策略）
        mcts_games = int(config['num_games'] * config['use_mcts_ratio'])
        random_games = config['num_games'] - mcts_games
        
        logger.info(f"生成MCTS数据: {mcts_games} 局")
        logger.info(f"生成智能随机数据: {random_games} 局")
        
        # 生成MCTS数据
        states_mcts, policies_mcts, values_mcts = data_generator.generate_training_data(
            num_games=mcts_games,
            strategy='mcts',
            save_path='data/mcts_data.npz'
        )
        
        # 生成智能随机数据
        states_random, policies_random, values_random = data_generator.generate_training_data(
            num_games=random_games,
            strategy='smart_random',
            save_path='data/random_data.npz'
        )
        
        # 合并数据
        import numpy as np
        states = np.concatenate([states_mcts, states_random], axis=0)
        policies = np.concatenate([policies_mcts, policies_random], axis=0)
        values = np.concatenate([values_mcts, values_random], axis=0)
        
        # 保存合并后的数据
        np.savez_compressed('data/training_data.npz', 
                          states=states, policies=policies, values=values)
        
        logger.info(f"生成训练数据完成: {len(states)} 个样本")
        
        # 2. 训练模型
        logger.info("步骤2: 训练神经网络模型")
        trainer = ModelTrainer(
            board_size=config['board_size'],
            channels=config['model_channels'],
            num_residual_blocks=config['num_residual_blocks'],
            learning_rate=config['learning_rate']
        )
        
        # 执行训练
        trainer.train(
            states=states,
            policies=policies,
            values=values,
            batch_size=config['batch_size'],
            max_time_seconds=config['max_training_time']
        )
        
        # 保存PyTorch模型
        model_path = 'models/gomoku_model.pth'
        trainer.save_model(model_path)
        logger.info(f"PyTorch模型已保存: {model_path}")
        
        # 3. 转换为KataGO格式
        logger.info("步骤3: 转换为KataGO兼容格式")
        converter = KataGoConverter()
        
        # 转换模型
        katago_model_path = 'output/gomoku_model.bin.gz'
        success = converter.convert_model(
            model=trainer.model,
            output_path=katago_model_path,
            model_name="GomokuFreestyle"
        )
        
        if success:
            logger.info(f"KataGO模型已生成: {katago_model_path}")
            
            # 验证模型大小
            is_valid, size_mb = converter.validate_model_size(
                katago_model_path, 
                config['target_model_size_mb']
            )
            
            if is_valid:
                logger.info(f"模型大小符合要求: {size_mb:.2f} MB")
            else:
                logger.warning(f"模型大小超出限制: {size_mb:.2f} MB")
        
        # 4. 生成配置文件和说明
        logger.info("步骤4: 生成KataGO配置和集成说明")
        
        # 创建KataGO配置文件
        config_path = converter.create_katago_config(
            model_path=katago_model_path,
            output_dir='output'
        )
        
        # 创建集成指南
        guide_path = converter.create_integration_guide('output')
        
        # 5. 模型性能评估
        logger.info("步骤5: 模型性能评估")
        
        # 使用部分数据进行评估
        eval_size = min(1000, len(states))
        eval_indices = np.random.choice(len(states), eval_size, replace=False)
        
        eval_states = states[eval_indices]
        eval_policies = policies[eval_indices]
        eval_values = values[eval_indices]
        
        policy_acc, value_acc = trainer.evaluate_model(eval_states, eval_policies, eval_values)
        
        logger.info(f"模型评估结果:")
        logger.info(f"  策略准确率: {policy_acc:.4f}")
        logger.info(f"  价值准确率: {value_acc:.4f}")
        
        # 6. 输出总结
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)
        print(f"生成的文件:")
        print(f"  - PyTorch模型: {model_path}")
        print(f"  - KataGO模型: {katago_model_path}")
        print(f"  - 配置文件: {config_path}")
        print(f"  - 集成指南: {guide_path}")
        print(f"  - 训练日志: training.log")
        
        if success:
            print(f"\n模型信息:")
            print(f"  - 模型大小: {size_mb:.2f} MB")
            print(f"  - 策略准确率: {policy_acc:.4f}")
            print(f"  - 价值准确率: {value_acc:.4f}")
            print(f"  - 训练样本数: {len(states)}")
            print("模型已准备好在KataGO 1.16.3中使用!")
        
        print("\n使用说明:")
        print("1. 查看 output/katago_integration_guide.md 了解详细的集成说明")
        print("2. 将 gomoku_model.bin.gz 和 katago_config.cfg 复制到KataGO目录")
        print("3. 运行: ./katago gtp -config katago_config.cfg")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        print(f"错误: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()