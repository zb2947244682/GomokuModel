# GomokuModel - 五子棋预训练模型

## 项目简介

这是一个专为KataGO 1.16.3设计的五子棋预训练模型项目。使用PyTorch框架训练CNN神经网络，生成与KataGO兼容的bin.gz格式模型文件。

## 功能特点

- 🎯 专为五子棋优化的CNN架构
- 🚀 CPU优化，10分钟快速训练
- 📦 生成KataGO兼容的bin.gz模型
- 🎲 高质量训练数据自动生成
- 🔄 数据增强（旋转、翻转）
- 🧠 简化MCTS自我对弈

## 系统要求

- Python 3.8+
- Windows系统
- 8GB+ RAM
- CPU: i5-10400或同等性能

## 安装步骤

1. 克隆项目到本地
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 运行训练：
   ```bash
   python main.py
   ```

## 输出文件

训练完成后，在`output/models/`目录下会生成：
- `gomoku_model.pth` - PyTorch模型文件
- `gomoku_model.bin.gz` - KataGO兼容模型文件

## KataGO集成说明

1. 将`gomoku_model.bin.gz`复制到KataGO的models目录
2. 修改KataGO配置文件，指向新模型
3. 启动KataGO即可使用五子棋模型

## 项目结构

```
GomokuModel/
├── src/                 # 源代码
│   ├── game/           # 游戏逻辑
│   ├── model/          # 神经网络
│   ├── data/           # 数据处理
│   └── utils/          # 工具函数
├── main.py             # 主程序
├── requirements.txt    # 依赖包
└── output/            # 输出目录
```

## 技术细节

- **棋盘大小**：15x15
- **输入格式**：15x15x3（黑子、白子、当前玩家）
- **输出格式**：策略头（225个位置概率）+ 价值头（胜负概率）
- **训练数据**：10000局高质量对局
- **模型大小**：约100MB

## 许可证

MIT License