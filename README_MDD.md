# Brain-JEPA MDD分类

本项目使用Brain-JEPA (Brain Joint Embedding Predictive Architecture)模型进行重性抑郁障碍(MDD)的分类任务。该项目基于原始的Brain-JEPA框架，针对MDD分类任务进行了适配和扩展。

## 项目结构

```
Brain-JEPA/
├── configs/
│   ├── mdd_classification.yaml     # MDD分类任务配置文件
│   └── downstream/
│       └── mdd_finetune.yaml       # MDD下游任务微调配置
├── downstream_tasks/
│   └── utils/
│       └── load_mdd.py             # MDD数据集加载工具
├── src/
│   └── models/
│       └── vision_transformer.py   # 修改后的ViT模型（兼容CUDA 11.6）
├── train_mdd.py                    # MDD分类训练主程序
├── predict_mdd.py                  # MDD分类预测主程序
├── run_mdd_train.sh                # MDD训练运行脚本
├── run_mdd_predict.sh              # MDD预测运行脚本
└── requirement.txt                 # 依赖项（已修改兼容CUDA 11.6）
```

## 环境准备

本项目已被修改以兼容CUDA 11.6环境。请先安装依赖：

```bash
pip install -r requirement.txt
```

主要依赖项包括：
- PyTorch 1.13.1+cu116
- torchvision 0.14.1+cu116
- torchaudio 0.13.1
- numpy
- scipy
- scikit-learn
- h5py
- pandas
- PyYAML

## 数据集

本项目使用ROI时间序列数据进行MDD分类。数据集应该包含：
- MDD患者的ROI时间序列数据
- 健康对照组的ROI时间序列数据

默认数据路径为：`/home/chentingyu/BrainDiagnostics/demo/ROISignals_FunImgARCWF`

## 训练模型

使用以下命令训练MDD分类模型：

```bash
./run_mdd_train.sh [参数]
```

可用参数：
- `--data_dir`：数据目录路径
- `--config_file`：配置文件路径
- `--output_dir`：输出目录
- `--batch_size`：批次大小
- `--epochs`：训练轮数
- `--lr`：学习率

示例：
```bash
./run_mdd_train.sh --batch_size 32 --epochs 100 --lr 0.0005
```

## 预测

训练完成后，使用以下命令进行预测：

```bash
./run_mdd_predict.sh <模型路径> [数据目录] [参数]
```

可用参数：
- `--output_dir`：输出目录
- `--batch_size`：批次大小
- `--mode`：预测模式（test、val 或 all）

示例：
```bash
./run_mdd_predict.sh ./output_dirs/mdd_classification/final_model.pth --mode all
```

## 模型架构

本项目使用Vision Transformer (ViT)作为基础模型，通过以下方式适配MDD分类任务：

1. 输入层接受单通道ROI时间序列数据
2. 输出层调整为二分类（MDD和对照组）
3. 添加了兼容CUDA 11.6的注意力机制

模型参数通过配置文件`configs/mdd_classification.yaml`控制。

## 结果评估

预测结果将包括以下指标：
- 准确率 (Accuracy)
- AUC值
- 分类报告（精确率、召回率、F1分数）
- 混淆矩阵

结果将保存在指定的输出目录中。

## 注意事项

1. 本项目已修改以兼容CUDA 11.6环境，移除了依赖flash_attn
2. 为确保结果可复现，预测时会使用固定的随机种子(42)
3. 训练脚本将自动创建所需的目录结构
4. 默认使用单GPU训练，可通过修改`CUDA_VISIBLE_DEVICES`环境变量控制使用的GPU

## 引用

如果您使用本项目，请引用以下论文：

```
@article{zhuang2023jepa,
  title={Self-supervised learning from images with a joint-embedding predictive architecture},
  author={Zhuang, Mahmoud Assran and Mathilde Caron and Ishan Misra and Pascal Vincent and Yann LeCun and Nicolas Ballas and Michael Rabbat},
  journal={arXiv preprint arXiv:2302.00965},
  year={2023}
}
``` 