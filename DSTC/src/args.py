import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # 文件路径相关
    parser.add_argument('--root_path', type=str, default="../data")
    parser.add_argument('--data_path', type=str, default="TUM_5_FOLD_DFT321_GN_DATA/Partition_1234")
    parser.add_argument('--json_name', type=str, default="Partition_1234_norm_DFT321_data_len64.json")
    parser.add_argument('--val_data_path', type=str, default="TUM_5_FOLD_DFT321_GN_DATA/Partition_5")
    parser.add_argument('--val_json_name', type=str, default="Partition_5_norm_DFT321_data_len64.json")
    # 训练数据相关参数
    parser.add_argument('--target_sample_hz', type=int, default=2_800)
    parser.add_argument('--data_length', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    # 训练过程相关参数（重要）
    parser.add_argument('--num_train_steps', type=int, default=20_000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--grad_accum_every', type=int, default=4)
    # 训练过程保存
    parser.add_argument('--save_model_every', type=int, default=100)
    parser.add_argument('--save_results_every', type=int, default=100)
    parser.add_argument('--log_losses_every', type=int, default=1)
    parser.add_argument('--apply_grad_penalty_every', type=int, default=4)
    # 优化器相关参数
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--discr_max_grad_norm', type=float, default=None)
    # EMA相关参数（别动）
    parser.add_argument('--ema_beta', type=float, default=0.995)
    parser.add_argument('--ema_update_after_step', type=int, default=500)
    parser.add_argument('--ema_update_every', type=int, default=10)

    args = parser.parse_args()
    return args
