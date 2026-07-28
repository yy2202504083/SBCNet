import os
import argparse
import re
from glob import glob
import prettytable as pt
import matplotlib.pyplot as plt
import numpy as np
from metrics import evaluator
from config import load_config


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ESCNet Evaluation Script")
    parser.add_argument(
        "--pred_root", type=str, help="Prediction root", default="preds"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Save directory for evaluation results",
        default="results",
    )
    parser.add_argument(
        "--check_integrity", type=bool, help="Check file integrity", default=True
    )
    parser.add_argument(
        "--model_lst",
        type=str,
        help="Comma-separated list of models/epochs to evaluate",
        default=None,
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file."
    )
    args = parser.parse_args()
    args.metrics = "+".join(["S", "MAE", "E", "F", "WF", "MSE"])
    args.model_folder = args.pred_root

    # 自动获取模型/Epoch 文件夹列表
    if args.model_lst is None:
        raw_folders = [
            d
            for d in os.listdir(args.model_folder)
            if os.path.isdir(os.path.join(args.model_folder, d))
        ]
        
        # 核心改进：为了防止出现文本排序错误（如 epoch_100 排在 epoch_60 前面），使用正则提取数字进行排序
        def get_epoch_num(folder_name):
            match = re.search(r"epoch_(\d+)", folder_name)
            return int(match.group(1)) if match else 9999
            
        args.model_lst = sorted(raw_folders, key=get_epoch_num)
    else:
        args.model_lst = args.model_lst.split(",")

    os.makedirs(args.save_dir, exist_ok=True)
    return args


def check_file_integrity(args, config):
    """Check the integrity of ground-truth and prediction files."""
    if args.check_integrity:
        print("Checking file integrity...")
        test_root = config.test_dir
        datasets = [d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))]
        for dataset_name in datasets:
            gt_dir = os.path.join(test_root, dataset_name, "GT")
            if not os.path.exists(gt_dir): continue
            for model_name in args.model_lst:
                pred_dir = os.path.join(args.pred_root, model_name, dataset_name)
                if not os.path.exists(pred_dir):
                    print(f"Warning: {model_name} missing on {dataset_name}")
                    continue
                if len(os.listdir(gt_dir)) != len(os.listdir(pred_dir)):
                    print(f"Error: Mismatch in {model_name} on {dataset_name}")
    else:
        print("Skipping integrity check.")


def get_raw_metrics(model_name, dataset_name, gt_paths, args):
    """获取原始数值，严格按照所需的 4 个指标顺序返回"""
    pred_paths = [os.path.join(args.pred_root, model_name, dataset_name, os.path.basename(p)) for p in gt_paths]
    em, sm, fm, mae, wfm, mba, biou = evaluator(
        gt_paths=gt_paths,
        pred_paths=pred_paths,
        metrics=args.metrics.split("+"),
    )
    # 对应顺序: Fw (wFmeasure), E (meanEm), S (Smeasure), M (MAE)
    return [
        wfm,
        em["curve"].mean(),
        sm,
        mae
    ]


def evaluate_all_and_summary(args, config):
    """遍历数据集，输出多数据集并排的宽表，并找出综合表现最好的 Epoch"""
    test_root = config.test_dir
    datasets = sorted([d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))])
    valid_datasets = [d for d in datasets if os.path.exists(os.path.join(test_root, d, "GT"))]
    print(f"Found valid datasets for table: {valid_datasets}")

    # 嵌套字典存储所有结果: { "epoch_60": { "CAMO": [Fw, E, S, M] } }
    all_results = {model: {} for model in args.model_lst}

    for dataset_name in valid_datasets:
        print(f"Evaluating dataset: {dataset_name}...")
        gt_dir = os.path.join(test_root, dataset_name, "GT")
        gt_paths = sorted(glob(os.path.join(gt_dir, "*")))
        for model_name in args.model_lst:
            try:
                metrics = get_raw_metrics(model_name, dataset_name, gt_paths, args)
                all_results[model_name][dataset_name] = metrics
            except Exception as e:
                print(f"Skip {model_name} on {dataset_name} due to error: {e}")
                all_results[model_name][dataset_name] = [0.0, 0.0, 0.0, 0.0]

    # --- 构造横向多数据集 PrettyTable ---
    tb = pt.PrettyTable()
    field_names = ["Methods"]
    for ds in valid_datasets:
        field_names.extend([f"{ds}_Fw", f"{ds}_E", f"{ds}_Sm", f"{ds}_Mae"])
    # 增加多数据集综合平均分表头
    field_names.extend(["Avg_Fw", "Avg_E", "Avg_Sm", "Avg_Mae"])
    tb.field_names = field_names

    # 用于寻找最优轮数的控制变量
    best_model_name = None
    best_score = -float("inf")  # 越高越好评估锚点
    best_metrics_summary = {}

    for model_name in args.model_lst:
        row = [model_name]
        
        # 累加各个数据集的指标以便计算平均分
        total_fw, total_e, total_sm, total_mae = 0.0, 0.0, 0.0, 0.0
        count_ds = len(valid_datasets)

        for ds in valid_datasets:
            metrics = all_results[model_name].get(ds, [0.0, 0.0, 0.0, 0.0])
            total_fw += metrics[0]
            total_e += metrics[1]
            total_sm += metrics[2]
            total_mae += metrics[3]

            formatted_metrics = [f"{v:.3f}" for v in metrics]
            row.extend(formatted_metrics)

        # 计算该 Epoch 在所有数据集上的平均指标数值
        avg_fw = total_fw / count_ds if count_ds > 0 else 0.0
        avg_e = total_e / count_ds if count_ds > 0 else 0.0
        avg_sm = total_sm / count_ds if count_ds > 0 else 0.0
        avg_mae = total_mae / count_ds if count_ds > 0 else 0.0

        row.extend([f"{avg_fw:.3f}", f"{avg_e:.3f}", f"{avg_sm:.3f}", f"{avg_mae:.3f}"])
        tb.add_row(row)

        # =====================================================
        # 核心新增：最优轮数（Best Epoch）自动决策算法
        # 计算综合宏得分 (Avg_Fw + Avg_E + Avg_Sm - Avg_Mae) 
        # =====================================================
        macro_score = avg_fw + avg_e + avg_sm - avg_mae
        if macro_score > best_score:
            best_score = macro_score
            best_model_name = model_name
            best_metrics_summary = {
                "Avg_Fw": avg_fw,
                "Avg_E": avg_e,
                "Avg_Sm": avg_sm,
                "Avg_Mae": avg_mae
            }

    # 终端输出宽表
    print("\n" + "=" * 100)
    print("  MULTI-DATASET BATCH EVALUATION RESULTS & OVERALL AVERAGE")
    print("=" * 100)
    print(tb)
    print("=" * 100)

    # 打印最终筛选出的黄金胜出轮数
    if best_model_name:
        print("\n" + "🌟" * 25)
        print(f" 【🏆 CRITICAL PROMPT: BEST PERFORMANCE EPOCH DETECTED 🏆】")
        print(f"  Best Validation Epoch :  {best_model_name}")
        print(f"  Comprehensive Scores  :  "
              f"Avg_Fw: {best_metrics_summary['Avg_Fw']:.3f} | "
              f"Avg_E : {best_metrics_summary['Avg_E']:.3f} | "
              f"Avg_Sm: {best_metrics_summary['Avg_Sm']:.3f} | "
              f"Avg_Mae: {best_metrics_summary['Avg_Mae']:.3f}")
        print("🌟" * 25 + "\n")

    # 保存日志
    save_path = os.path.join(args.save_dir, "multi_dataset_results.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(str(tb))
        if best_model_name:
            f.write(f"\n\n>>> BEST PERFORMING CHECKPOINT: {best_model_name}\n")
            f.write(f"Metrics -> Avg_Fw: {best_metrics_summary['Avg_Fw']:.3f} | Avg_E: {best_metrics_summary['Avg_E']:.3f} | Avg_Sm: {best_metrics_summary['Avg_Sm']:.3f} | Avg_Mae: {best_metrics_summary['Avg_Mae']:.3f}\n")
            
    print(f"Results successfully dumped to: {save_path}")


if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config)
    check_file_integrity(args, config)
    evaluate_all_and_summary(args, config)
