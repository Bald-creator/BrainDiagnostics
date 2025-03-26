#!/bin/bash

# MDD分类训练脚本

# 创建必要的目录
mkdir -p output_dirs/mdd_classification/logs

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=0

# 默认参数
DATA_DIR="/home/chentingyu/BrainDiagnostics/demo/ROISignals_FunImgARCWF"
CONFIG_FILE="./configs/mdd_classification.yaml"
OUTPUT_DIR="./output_dirs/mdd_classification"
BATCH_SIZE=16
EPOCHS=50
LR=0.001

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --config_file)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 打印训练配置
echo "=== MDD Classification Training ==="
echo "数据目录: $DATA_DIR"
echo "配置文件: $CONFIG_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "批次大小: $BATCH_SIZE"
echo "训练轮数: $EPOCHS"
echo "学习率: $LR"
echo "=================================="

# 开始训练
echo "开始MDD分类训练..."
python train_mdd.py \
  --data_dir "$DATA_DIR" \
  --config_file "$CONFIG_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR"

# 检查训练结果
if [ $? -eq 0 ]; then
  echo "MDD分类训练完成！"
  echo "模型保存在: $OUTPUT_DIR"
else
  echo "MDD分类训练失败！"
  exit 1
fi 