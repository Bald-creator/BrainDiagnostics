#!/bin/bash

# MDD分类预测脚本

# 检查参数
if [ $# -lt 1 ]; then
  echo "用法: $0 <模型路径> [数据目录]"
  echo "示例: $0 ./output_dirs/mdd_classification/final_model.pth"
  exit 1
fi

MODEL_PATH=$1

# 创建必要的目录
mkdir -p output_dirs/mdd_prediction

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=0

# 默认参数
DATA_DIR="/home/chentingyu/BrainDiagnostics/demo/ROISignals_FunImgARCWF"
OUTPUT_DIR="./output_dirs/mdd_prediction"
BATCH_SIZE=16
MODE="test"

# 如果提供了第二个参数，使用它作为数据目录
if [ $# -ge 2 ]; then
  DATA_DIR=$2
fi

# 解析命令行参数
shift
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_dir)
      DATA_DIR="$2"
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
    --mode)
      MODE="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 打印预测配置
echo "=== MDD Classification Prediction ==="
echo "模型路径: $MODEL_PATH"
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "批次大小: $BATCH_SIZE"
echo "预测模式: $MODE"
echo "===================================="

# 开始预测
echo "开始MDD分类预测..."
python predict_mdd.py \
  --model_path "$MODEL_PATH" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --mode "$MODE"

# 检查预测结果
if [ $? -eq 0 ]; then
  echo "MDD分类预测完成！"
  echo "结果保存在: $OUTPUT_DIR/results_${MODE}.json"
else
  echo "MDD分类预测失败！"
  exit 1
fi 