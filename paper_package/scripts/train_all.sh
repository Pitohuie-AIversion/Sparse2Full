#!/bin/bash
# һ��ѵ���ű�
# ʹ�÷���: bash train_all.sh

set -e

echo "��ʼѵ������ģ��..."

# ���û���
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# ѵ�������б�
configs=(
    "configs/sr_darcy_swinunet.yaml"
    "configs/sr_darcy_unet.yaml"
    "configs/crop_cfd_hybrid.yaml"
)

# �����б�
seeds=(42 123 456)

# ѵ����������
for config in "${configs[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "ѵ������: $config, ����: $seed"
        python tools/train.py --config $config --seed $seed
    done
done

echo "ѵ����ɣ�"
