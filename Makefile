# PDEBench稀疏观测重建系统 Makefile
# 提供项目管理、测试、训练、评估等自动化命令

# 项目配置
PROJECT_NAME = Sparse2Full
PYTHON = python
PIP = pip
CONDA = conda

# 目录配置
DATA_DIR = data
RUNS_DIR = runs
PAPER_DIR = paper_package
TESTS_DIR = tests
CONFIGS_DIR = configs

# 默认配置
DEFAULT_CONFIG = configs/sr_swin_unet_2x.yaml
DEFAULT_GPU = 0
DEFAULT_SEED = 2025

# 颜色输出
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

.PHONY: help install install-dev setup-env clean test lint format \
        train eval benchmark visualize paper-package \
        data-download data-prepare check-consistency \
        docker-build docker-run

# 默认目标
all: help

# 帮助信息
help:
	@echo "$(BLUE)PDEBench稀疏观测重建系统 - 自动化命令$(NC)"
	@echo ""
	@echo "$(GREEN)环境管理:$(NC)"
	@echo "  install          安装项目依赖"
	@echo "  install-dev      安装开发依赖"
	@echo "  setup-env        设置conda环境"
	@echo "  clean            清理临时文件"
	@echo ""
	@echo "$(GREEN)代码质量:$(NC)"
	@echo "  test             运行所有测试"
	@echo "  test-unit        运行单元测试"
	@echo "  test-e2e         运行端到端测试"
	@echo "  lint             代码风格检查"
	@echo "  format           代码格式化"
	@echo "  check-consistency 检查观测算子一致性"
	@echo ""
	@echo "$(GREEN)数据管理:$(NC)"
	@echo "  data-download    下载PDEBench数据集"
	@echo "  data-prepare     数据预处理和切分"
	@echo "  data-stats       数据统计信息"
	@echo ""
	@echo "$(GREEN)模型训练:$(NC)"
	@echo "  train            单GPU训练 (默认配置)"
	@echo "  train-multi      多GPU分布式训练"
	@echo "  train-sr         超分辨率任务训练"
	@echo "  train-crop       裁剪重建任务训练"
	@echo ""
	@echo "$(GREEN)模型评估:$(NC)"
	@echo "  eval             模型评估"
	@echo "  benchmark        模型基准测试"
	@echo "  visualize        生成可视化结果"
	@echo ""
	@echo "$(GREEN)论文材料:$(NC)"
	@echo "  paper-package    生成论文材料包"
	@echo "  paper-anonymous  生成匿名化论文包"
	@echo ""
	@echo "$(GREEN)Docker支持:$(NC)"
	@echo "  docker-build     构建Docker镜像"
	@echo "  docker-run       运行Docker容器"
	@echo ""
	@echo "$(YELLOW)使用示例:$(NC)"
	@echo "  make install                    # 安装依赖"
	@echo "  make train CONFIG=configs/sr_swin_unet_4x.yaml GPU=1"
	@echo "  make eval CHECKPOINT=runs/best.pth"
	@echo "  make benchmark MODELS=swin_unet,hybrid,fno"

# ============================================================================
# 环境管理
# ============================================================================

install:
	@echo "$(GREEN)安装项目依赖...$(NC)"
	$(PIP) install -r requirements.txt

install-dev:
	@echo "$(GREEN)安装开发依赖...$(NC)"
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

setup-env:
	@echo "$(GREEN)设置conda环境...$(NC)"
	$(CONDA) env create -f environment.yml
	@echo "$(YELLOW)请运行: conda activate sparse2full$(NC)"

clean:
	@echo "$(GREEN)清理临时文件...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

# ============================================================================
# 代码质量
# ============================================================================

test:
	@echo "$(GREEN)运行所有测试...$(NC)"
	pytest $(TESTS_DIR)/ -v --cov=. --cov-report=html --cov-report=term

test-unit:
	@echo "$(GREEN)运行单元测试...$(NC)"
	pytest $(TESTS_DIR)/test_*.py -v -k "not e2e"

test-e2e:
	@echo "$(GREEN)运行端到端测试...$(NC)"
	$(PYTHON) $(TESTS_DIR)/test_e2e_simple.py

lint:
	@echo "$(GREEN)代码风格检查...$(NC)"
	ruff check .
	mypy . --strict

format:
	@echo "$(GREEN)代码格式化...$(NC)"
	black .
	isort .
	ruff check . --fix

check-consistency:
	@echo "$(GREEN)检查观测算子一致性...$(NC)"
	$(PYTHON) tools/check_dc_equivalence.py --config $(DEFAULT_CONFIG)

# ============================================================================
# 数据管理
# ============================================================================

data-download:
	@echo "$(GREEN)下载PDEBench数据集...$(NC)"
	mkdir -p $(DATA_DIR)
	$(PYTHON) tools/download_data.py --dataset pdebench --output_dir $(DATA_DIR)

data-prepare:
	@echo "$(GREEN)数据预处理和切分...$(NC)"
	$(PYTHON) tools/prepare_data.py --data_root $(DATA_DIR)/pdebench --output_dir $(DATA_DIR)/processed

data-stats:
	@echo "$(GREEN)生成数据统计信息...$(NC)"
	$(PYTHON) tools/data_statistics.py --data_dir $(DATA_DIR)/processed

# ============================================================================
# 模型训练
# ============================================================================

# 变量设置
CONFIG ?= $(DEFAULT_CONFIG)
GPU ?= $(DEFAULT_GPU)
SEED ?= $(DEFAULT_SEED)
BATCH_SIZE ?= 16
EPOCHS ?= 200

train:
	@echo "$(GREEN)开始单GPU训练...$(NC)"
	@echo "配置文件: $(CONFIG)"
	@echo "GPU: $(GPU)"
	@echo "随机种子: $(SEED)"
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) tools/train.py \
		--config $(CONFIG) \
		--seed $(SEED) \
		--batch_size $(BATCH_SIZE) \
		--epochs $(EPOCHS)

train-multi:
	@echo "$(GREEN)开始多GPU分布式训练...$(NC)"
	$(PYTHON) -m torch.distributed.launch --nproc_per_node=4 \
		tools/train.py --config $(CONFIG) --distributed --seed $(SEED)

train-sr:
	@echo "$(GREEN)训练超分辨率模型...$(NC)"
	make train CONFIG=configs/sr_swin_unet_2x.yaml
	make train CONFIG=configs/sr_swin_unet_4x.yaml
	make train CONFIG=configs/sr_hybrid_2x.yaml

train-crop:
	@echo "$(GREEN)训练裁剪重建模型...$(NC)"
	make train CONFIG=configs/crop_swin_unet_40p.yaml
	make train CONFIG=configs/crop_hybrid_40p.yaml
	make train CONFIG=configs/crop_fno_40p.yaml

# ============================================================================
# 模型评估
# ============================================================================

CHECKPOINT ?= runs/latest/best.pth
MODELS ?= swin_unet,hybrid,fno
TASKS ?= sr,crop
SEEDS ?= 2025,2026,2027

eval:
	@echo "$(GREEN)模型评估...$(NC)"
	@echo "检查点: $(CHECKPOINT)"
	$(PYTHON) tools/eval.py --config $(CONFIG) --checkpoint $(CHECKPOINT)

benchmark:
	@echo "$(GREEN)模型基准测试...$(NC)"
	@echo "模型: $(MODELS)"
	@echo "任务: $(TASKS)"
	@echo "种子: $(SEEDS)"
	$(PYTHON) tools/benchmark_models.py \
		--models $(MODELS) \
		--tasks $(TASKS) \
		--seeds $(SEEDS) \
		--output_dir benchmarks/

visualize:
	@echo "$(GREEN)生成可视化结果...$(NC)"
	$(PYTHON) tools/visualize_results.py \
		--runs_dir $(RUNS_DIR) \
		--output_dir visualizations/

# ============================================================================
# 论文材料
# ============================================================================

paper-package:
	@echo "$(GREEN)生成论文材料包...$(NC)"
	$(PYTHON) tools/generate_paper_package.py \
		--runs_dir $(RUNS_DIR) \
		--output_dir $(PAPER_DIR)

paper-anonymous:
	@echo "$(GREEN)生成匿名化论文包...$(NC)"
	$(PYTHON) tools/generate_paper_package.py \
		--runs_dir $(RUNS_DIR) \
		--output_dir $(PAPER_DIR) \
		--anonymous

# ============================================================================
# Docker支持
# ============================================================================

DOCKER_IMAGE = pdebench/sparse2full
DOCKER_TAG = latest

docker-build:
	@echo "$(GREEN)构建Docker镜像...$(NC)"
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run:
	@echo "$(GREEN)运行Docker容器...$(NC)"
	docker run --gpus all -it --rm \
		-v $(PWD):/workspace \
		-v $(PWD)/data:/workspace/data \
		-v $(PWD)/runs:/workspace/runs \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

# ============================================================================
# 快速命令
# ============================================================================

# 快速开始 - 完整流程
quickstart: install data-download data-prepare train-sr eval

# 开发环境设置
dev-setup: install-dev setup-env test lint

# CI/CD流程
ci: lint test check-consistency

# 发布准备
release: clean test lint benchmark paper-package

# ============================================================================
# 实验管理
# ============================================================================

# 清理实验结果
clean-runs:
	@echo "$(YELLOW)清理实验结果...$(NC)"
	@read -p "确认删除所有实验结果? [y/N] " confirm && [ "$$confirm" = "y" ]
	rm -rf $(RUNS_DIR)/*

# 备份实验结果
backup-runs:
	@echo "$(GREEN)备份实验结果...$(NC)"
	tar -czf runs_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz $(RUNS_DIR)/

# 列出实验
list-runs:
	@echo "$(GREEN)实验列表:$(NC)"
	@ls -la $(RUNS_DIR)/ | grep "^d" | awk '{print $$9}' | grep -v "^\." | sort

# ============================================================================
# 调试和诊断
# ============================================================================

debug-train:
	@echo "$(GREEN)调试模式训练...$(NC)"
	$(PYTHON) tools/train.py --config $(CONFIG) --debug --max_samples 100

debug-eval:
	@echo "$(GREEN)调试模式评估...$(NC)"
	$(PYTHON) tools/eval.py --config $(CONFIG) --checkpoint $(CHECKPOINT) --debug

check-env:
	@echo "$(GREEN)检查环境配置...$(NC)"
	@echo "Python版本: $(shell $(PYTHON) --version)"
	@echo "PyTorch版本: $(shell $(PYTHON) -c 'import torch; print(torch.__version__)')"
	@echo "CUDA可用: $(shell $(PYTHON) -c 'import torch; print(torch.cuda.is_available())')"
	@echo "GPU数量: $(shell $(PYTHON) -c 'import torch; print(torch.cuda.device_count())')"

# ============================================================================
# 性能分析
# ============================================================================

profile-train:
	@echo "$(GREEN)性能分析 - 训练...$(NC)"
	$(PYTHON) -m cProfile -o train_profile.prof tools/train.py --config $(CONFIG) --max_epochs 1
	$(PYTHON) -c "import pstats; pstats.Stats('train_profile.prof').sort_stats('cumulative').print_stats(20)"

profile-eval:
	@echo "$(GREEN)性能分析 - 评估...$(NC)"
	$(PYTHON) -m cProfile -o eval_profile.prof tools/eval.py --config $(CONFIG) --checkpoint $(CHECKPOINT)
	$(PYTHON) -c "import pstats; pstats.Stats('eval_profile.prof').sort_stats('cumulative').print_stats(20)"

# ============================================================================
# 文档生成
# ============================================================================

docs:
	@echo "$(GREEN)生成API文档...$(NC)"
	sphinx-build -b html docs/ docs/_build/html

docs-serve:
	@echo "$(GREEN)启动文档服务器...$(NC)"
	cd docs/_build/html && $(PYTHON) -m http.server 8000

# ============================================================================
# 版本管理
# ============================================================================

version:
	@echo "$(GREEN)项目版本信息:$(NC)"
	@echo "Git提交: $(shell git rev-parse HEAD)"
	@echo "Git分支: $(shell git branch --show-current)"
	@echo "最后修改: $(shell git log -1 --format=%cd)"

tag:
	@echo "$(GREEN)创建版本标签...$(NC)"
	@read -p "输入版本号 (例: v1.0.0): " version && \
	git tag -a $$version -m "Release $$version" && \
	git push origin $$version

# ============================================================================
# 错误处理
# ============================================================================

# 检查必要的目录
check-dirs:
	@mkdir -p $(DATA_DIR) $(RUNS_DIR) $(PAPER_DIR)

# 检查Python依赖
check-deps:
	@$(PYTHON) -c "import torch, numpy, matplotlib, seaborn" || \
		(echo "$(RED)缺少必要依赖，请运行: make install$(NC)" && exit 1)

# 在主要命令前检查环境
train eval benchmark: check-dirs check-deps