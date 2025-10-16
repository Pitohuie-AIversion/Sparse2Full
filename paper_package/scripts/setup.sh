#!/bin/bash
# ������װ�ű�
# ʹ�÷���: bash setup.sh

set -e

echo "��װPDEBenchϡ��۲��ؽ�ϵͳ����..."

# ����conda����
conda create -n pdebench python=3.10 -y
conda activate pdebench

# ��װPyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# ��װ��������
pip install -r requirements.txt

echo "������װ��ɣ�"
echo "ʹ�� 'conda activate pdebench' �����"
