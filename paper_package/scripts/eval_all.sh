#!/bin/bash
# һ�������ű�
# ʹ�÷���: bash eval_all.sh

set -e

echo "��ʼ��������ģ��..."

# ��������ʵ��
python tools/eval.py --runs_dir runs --output_dir paper_package/metrics

# �������Ĳ��ϰ�
python tools/generate_paper_package.py --runs_dir runs --output_dir paper_package

echo "������ɣ�"
