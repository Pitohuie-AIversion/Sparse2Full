"""简化的集成测试脚本

验证时序NAR架构的核心功能和兼容性。
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_temporal_modules():
    """测试时序模块"""
    print("测试时序模块...")
    
    try:
        from models.temporal_block import TemporalConv1D, FiLMTemporalBlock, create_temporal_module
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        B, T, C, H, W = 2, 4, 1, 64, 64
        x_seq = torch.randn(B, T, C, H, W, device=device)
        
        # 测试TemporalConv1D
        temporal_conv = TemporalConv1D(c_in=C, c_out=C, k=3, causal=True).to(device)
        out_conv = temporal_conv(x_seq)
        assert out_conv.shape == (B, C, H, W), f"TemporalConv1D输出形状错误: {out_conv.shape}"
        
        # 测试FiLMTemporalBlock
        temporal_film = FiLMTemporalBlock(c_in=C, c_out=C).to(device)
        out_film = temporal_film(x_seq)
        assert out_film.shape == (B, C, H, W), f"FiLMTemporalBlock输出形状错误: {out_film.shape}"
        
        # 测试工厂函数
        temporal_factory = create_temporal_module('conv1d', c_in=C, c_out=C, k=3).to(device)
        out_factory = temporal_factory(x_seq)
        assert out_factory.shape == (B, C, H, W), f"工厂时序模块输出形状错误: {out_factory.shape}"
        
        print("✓ 时序模块测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 时序模块测试失败: {e}")
        return False


def test_query_heads():
    """测试NAR查询头"""
    print("测试NAR查询头...")
    
    try:
        from models.decoder.query_head import TimeQueryHead, CrossAttentionQueryHead, create_query_head
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        B, T_out, d_model, H, W = 2, 3, 96, 16, 16
        C = 1
        
        # 生成memory特征
        memory = torch.randn(B, d_model, H, W, device=device)
        
        # 测试TimeQueryHead
        time_head = TimeQueryHead(d_model=d_model, c_out=C, max_timesteps=32).to(device)
        out_time = time_head(memory, T_out)
        assert out_time.shape == (B, T_out, C, H, W), f"TimeQueryHead输出形状错误: {out_time.shape}"
        
        # 测试CrossAttentionQueryHead
        cross_head = CrossAttentionQueryHead(d_model=d_model, c_out=C, num_heads=8, max_timesteps=32).to(device)
        out_cross = cross_head(memory, T_out)
        assert out_cross.shape == (B, T_out, C, H, W), f"CrossAttentionQueryHead输出形状错误: {out_cross.shape}"
        
        # 测试工厂函数
        head_factory = create_query_head('simple', d_model=d_model, c_out=C).to(device)
        out_factory = head_factory(memory, T_out)
        assert out_factory.shape == (B, T_out, C, H, W), f"工厂查询头输出形状错误: {out_factory.shape}"
        
        print("✓ NAR查询头测试通过")
        return True
        
    except Exception as e:
        print(f"✗ NAR查询头测试失败: {e}")
        return False


def test_swin_temporal():
    """测试Swin时序包装器"""
    print("测试Swin时序包装器...")
    
    try:
        from models.wrappers.swin_temporal import SwinTemporal, SwinTemporalNAR
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        B, T_in, T_out, C, H, W = 2, 4, 3, 1, 64, 64
        x_seq = torch.randn(B, T_in, C, H, W, device=device)
        teacher_seq = torch.randn(B, T_out, C, H, W, device=device)
        
        # SwinUNet基础配置
        base_kwargs = {
            'in_channels': C,
            'out_channels': C,
            'img_size': H,
            'patch_size': 4,
            'embed_dim': 96,
            'depths': [2, 2, 2, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 8
        }
        
        # 时序配置
        temporal_cfg = {
            'enabled': True,
            'type': 'conv1d',
            'c_out': C,
            'k': 3,
            'causal': True
        }
        
        # 测试SwinTemporal
        swin_temporal = SwinTemporal(base_kwargs, temporal_cfg).to(device)
        
        # 单帧输入
        x_single = x_seq[:, -1]  # (B, C, H, W)
        out_single = swin_temporal(x_single)
        assert out_single.shape == (B, C, H, W), f"SwinTemporal单帧输出形状错误: {out_single.shape}"
        
        # 多帧输入
        out_multi = swin_temporal(x_seq)
        assert out_multi.shape == (B, C, H, W), f"SwinTemporal多帧输出形状错误: {out_multi.shape}"
        
        print("✓ Swin时序包装器测试通过")
        return True
        
    except Exception as e:
        print(f"✗ Swin时序包装器测试失败: {e}")
        return False


def test_temporal_nar_integration():
    """测试时序NAR完整集成"""
    print("测试时序NAR完整集成...")
    
    try:
        from models.wrappers.ar_nar_wrapper import ARNARWrapper, ARNAROutput
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        B, T_in, T_out, C, H, W = 2, 4, 3, 1, 32, 32  # 使用较小尺寸加速测试
        x_seq = torch.randn(B, T_in, C, H, W, device=device)
        target_seq = torch.randn(B, T_out, C, H, W, device=device)
        teacher_seq = torch.randn(B, T_out, C, H, W, device=device)
        
        # 配置
        model_config = {
            'base_kwargs': {
                'in_channels': C,
                'out_channels': C,
                'img_size': H,
                'patch_size': 4,
                'embed_dim': 64,
                'depths': [2, 2],
                'num_heads': [2, 4],
                'window_size': 4,
                'mlp_ratio': 2.0,
                'drop_rate': 0.0,
                'attn_drop_rate': 0.0,
                'drop_path_rate': 0.0,
            },
            'temporal': {
                'T_in': T_in,
                'T_out': T_out,
                'temporal_embed_dim': 32,
                'temporal_depth': 2,
                'temporal_heads': 2,
                'use_temporal_pe': True,
            },
            'nar': {
                'query_dim': 64,
                'num_queries': T_out,  # 使用T_out而不是硬编码的16
                'num_layers': 2,
                'num_heads': 4,
                'dropout': 0.0,
            },
            'ar': {
                'detach_rollout': True,
                'scheduled_sampling': False,
                'sampling_schedule': {
                    'start_prob': 0.0,
                    'end_prob': 0.5,
                    'schedule_type': 'linear'
                }
            },
            'use_ar': True,
            'use_nar': True,
        }
        
        loss_config = {
            'ar_weight': 1.0,
            'nar_weight': 1.0,
            'ar_weight_schedule': 'constant',
            'nar_weight_schedule': 'constant'
        }
        
        training_config = {
            'inference_mode': 'nar',
            'total_epochs': 100,
            'enable_monitoring': False,
            'monitoring_interval': 100,
            'ensemble_weight': 0.5
        }
        
        # 创建包装器
        wrapper = ARNARWrapper(model_config, loss_config, training_config).to(device)
        
        # 1. 训练前向传播测试
        wrapper.train()
        train_output = wrapper(
            x_seq=x_seq,
            T_out=T_out,
            teacher_seq=teacher_seq,
            compute_loss=True,
            target_seq=target_seq
        )
        
        # 检查训练输出
        if hasattr(train_output, 'total_loss') and train_output.total_loss is not None:
            print(f"✓ 训练损失: {train_output.total_loss.item():.6f}")
            if hasattr(train_output, 'ar_pred') and train_output.ar_pred is not None:
                print(f"✓ AR预测形状: {train_output.ar_pred.shape}")
            if hasattr(train_output, 'nar_pred') and train_output.nar_pred is not None:
                print(f"✓ NAR预测形状: {train_output.nar_pred.shape}")
        else:
            print("⚠️ 训练输出格式异常")
        
        # 2. 推理模式测试
        wrapper.eval()
        
        # AR推理
        wrapper.set_inference_mode('ar')
        ar_pred = wrapper(x_seq=x_seq, T_out=T_out, compute_loss=False)
        assert ar_pred.shape == (B, T_out, C, H, W), f"AR推理输出形状错误: {ar_pred.shape}"
        print(f"✓ AR推理形状: {ar_pred.shape}")
        
        # NAR推理
        wrapper.set_inference_mode('nar')
        nar_pred = wrapper(x_seq=x_seq, T_out=T_out, compute_loss=False)
        assert nar_pred.shape == (B, T_out, C, H, W), f"NAR推理输出形状错误: {nar_pred.shape}"
        print(f"✓ NAR推理形状: {nar_pred.shape}")
        
        # 集成推理
        wrapper.set_inference_mode('ensemble')
        ensemble_pred = wrapper(x_seq=x_seq, T_out=T_out, compute_loss=False)
        assert ensemble_pred.shape == (B, T_out, C, H, W), f"集成推理输出形状错误: {ensemble_pred.shape}"
        print(f"✓ 集成推理形状: {ensemble_pred.shape}")
        
        # 3. 性能指标测试
        ar_mse = torch.nn.functional.mse_loss(ar_pred, target_seq).item()
        nar_mse = torch.nn.functional.mse_loss(nar_pred, target_seq).item()
        ensemble_mse = torch.nn.functional.mse_loss(ensemble_pred, target_seq).item()
        
        print(f"✓ AR MSE: {ar_mse:.6f}")
        print(f"✓ NAR MSE: {nar_mse:.6f}")
        print(f"✓ 集成 MSE: {ensemble_mse:.6f}")
        
        # 4. 模型信息测试
        model_info = wrapper.get_model_info()
        print(f"✓ 模型参数数量: {model_info.get('total_parameters', 0):,}")
        print(f"✓ 推理模式: {model_info.get('inference_mode', 'unknown')}")
        
        print("✓ 时序NAR完整集成测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 时序NAR完整集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_temporal_nar_compatibility():
    """测试时序NAR与现有训练管线的兼容性"""
    print("测试时序NAR兼容性...")
    
    try:
        from models.wrappers.ar_nar_wrapper import ARNARWrapper
        import yaml
        from pathlib import Path
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 尝试加载配置文件
        config_path = Path("configs/temporal_nar_test.yaml")
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 创建模型
            model = ARNARWrapper(
                model_config=config['model'],
                loss_config=config['loss'],
                training_config=config['train']
            ).to(device)
            
            print("✓ 配置文件加载成功")
            print(f"✓ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
        else:
            print("⚠️ 配置文件不存在，跳过配置兼容性测试")
        
        # 测试与优化器的兼容性
        from models.wrappers.ar_nar_wrapper import ARNARWrapper
        
        model_config = {
            'base_kwargs': {'in_channels': 1, 'out_channels': 1, 'img_size': 32, 'patch_size': 4, 'embed_dim': 64, 'depths': [2], 'num_heads': [2], 'window_size': 4},
            'temporal': {'T_in': 4, 'T_out': 2, 'temporal_embed_dim': 32, 'temporal_depth': 1, 'temporal_heads': 2, 'use_temporal_pe': True},
            'nar': {'query_dim': 64, 'num_queries': 8, 'num_layers': 1, 'num_heads': 2, 'dropout': 0.0},
            'ar': {'detach_rollout': True, 'scheduled_sampling': False, 'sampling_schedule': {'start_prob': 0.0, 'end_prob': 0.5, 'schedule_type': 'linear'}},
            'use_ar': True, 'use_nar': True
        }
        
        loss_config = {'ar_weight': 1.0, 'nar_weight': 1.0, 'ar_weight_schedule': 'constant', 'nar_weight_schedule': 'constant'}
        training_config = {'inference_mode': 'nar', 'total_epochs': 100, 'enable_monitoring': False}
        
        model = ARNARWrapper(model_config, loss_config, training_config).to(device)
        
        # 测试优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        print("✓ 优化器兼容性测试通过")
        
        # 测试学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        print("✓ 学习率调度器兼容性测试通过")
        
        # 测试epoch设置
        model.set_epoch(10, 100)
        print("✓ Epoch设置兼容性测试通过")
        
        print("✓ 时序NAR兼容性测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 时序NAR兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ar_nar_wrapper():
    """测试AR-NAR双头包装器"""
    print("测试AR-NAR双头包装器...")
    
    try:
        from models.wrappers.ar_nar_wrapper import ARNARWrapper, ARNAROutput
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        B, T_in, T_out, C, H, W = 2, 4, 3, 1, 64, 64
        x_seq = torch.randn(B, T_in, C, H, W, device=device)
        target_seq = torch.randn(B, T_out, C, H, W, device=device)
        teacher_seq = torch.randn(B, T_out, C, H, W, device=device)
        
        # 配置
        model_config = {
            'base_kwargs': {
                'in_channels': C,
                'out_channels': C,
                'img_size': H,
                'patch_size': 4,
                'embed_dim': 96,
                'depths': [2, 2, 2, 2],
                'num_heads': [3, 6, 12, 24],
                'window_size': 8
            },
            'temporal': {
                'enabled': True,
                'type': 'conv1d',
                'c_out': C,
                'k': 3,
                'causal': True
            },
            'nar': {
                'head_type': 'simple',
                'd_model': 96,
                'max_timesteps': 32
            },
            'ar': {
                'detach_rollout': True,
                'scheduled_sampling': False
            },
            'use_ar': True,
            'use_nar': True
        }
        
        loss_config = {
            'ar_weight': 1.0,
            'nar_weight': 1.0,
            'ar_weight_schedule': 'constant',
            'nar_weight_schedule': 'constant'
        }
        
        training_config = {
            'inference_mode': 'nar',
            'total_epochs': 100,
            'enable_monitoring': False
        }
        
        # 创建包装器
        wrapper = ARNARWrapper(model_config, loss_config, training_config).to(device)
        
        # 训练前向传播
        wrapper.train()
        train_output = wrapper(
            x_seq=x_seq,
            T_out=T_out,
            teacher_seq=teacher_seq,
            compute_loss=True,
            target_seq=target_seq
        )
        
        assert isinstance(train_output, ARNAROutput), f"训练输出类型错误: {type(train_output)}"
        assert train_output.total_loss is not None, "总损失不应为None"
        
        # 推理前向传播
        wrapper.eval()
        with torch.no_grad():
            inference_output = wrapper(x_seq=x_seq, T_out=T_out, compute_loss=False)
            assert inference_output.shape == (B, T_out, C, H, W), f"推理输出形状错误: {inference_output.shape}"
        
        print("✓ AR-NAR双头包装器测试通过")
        return True
        
    except Exception as e:
        print(f"✗ AR-NAR双头包装器测试失败: {e}")
        return False


def test_ar_compatibility():
    """测试AR兼容性"""
    print("测试AR兼容性...")
    
    try:
        from models.ar.wrapper import ARWrapper
        from models.wrappers.ar_nar_wrapper import ARNARWrapper
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        B, T_in, T_out, C, H, W = 2, 4, 3, 1, 64, 64
        x_seq = torch.randn(B, T_in, C, H, W, device=device)
        
        # 创建AR包装器
        from models.swin_unet import SwinUNet
        
        base_model = SwinUNet(
            in_channels=C,
            out_channels=C,
            img_size=H,
            patch_size=4,
            embed_dim=96,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8
        )
        
        ar_wrapper = ARWrapper(
            single_frame_model=base_model,
            detach_rollout=True,
            scheduled_sampling=False
        ).to(device)
        
        # 推理测试
        ar_wrapper.eval()
        with torch.no_grad():
            ar_output = ar_wrapper(x_seq, T_out)
            assert ar_output.shape == (B, T_out, C, H, W), f"AR输出形状错误: {ar_output.shape}"
        
        # 创建AR-NAR包装器（仅AR模式）
        model_config = {
            'base_kwargs': {
                'in_channels': C,
                'out_channels': C,
                'img_size': H,
                'patch_size': 4,
                'embed_dim': 96,
                'depths': [2, 2, 2, 2],
                'num_heads': [3, 6, 12, 24],
                'window_size': 8
            },
            'temporal': {'enabled': False},
            'nar': {'head_type': 'simple', 'd_model': 96, 'max_timesteps': 32},
            'ar': {'detach_rollout': True, 'scheduled_sampling': False},
            'use_ar': True,
            'use_nar': False
        }
        
        loss_config = {'ar_weight': 1.0, 'nar_weight': 0.0}
        training_config = {'inference_mode': 'ar', 'total_epochs': 100, 'enable_monitoring': False}
        
        ar_nar_wrapper = ARNARWrapper(model_config, loss_config, training_config).to(device)
        
        # 推理测试
        ar_nar_wrapper.eval()
        with torch.no_grad():
            ar_nar_output = ar_nar_wrapper(x_seq=x_seq, T_out=T_out, compute_loss=False)
            assert ar_nar_output.shape == (B, T_out, C, H, W), f"AR-NAR输出形状错误: {ar_nar_output.shape}"
        
        print("✓ AR兼容性测试通过")
        return True
        
    except Exception as e:
        print(f"✗ AR兼容性测试失败: {e}")
        return False


def test_model_factory():
    """测试模型工厂函数"""
    print("测试模型工厂函数...")
    
    try:
        from models import create_model
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        B, T_in, T_out, C, H, W = 2, 4, 3, 1, 64, 64
        x_seq = torch.randn(B, T_in, C, H, W, device=device)
        
        # 测试通过工厂函数创建AR-NAR模型
        model_config = {
            'model_config': {
                'base_kwargs': {
                    'in_channels': C,
                    'out_channels': C,
                    'img_size': H,
                    'patch_size': 4,
                    'embed_dim': 96,
                    'depths': [2, 2, 2, 2],
                    'num_heads': [3, 6, 12, 24],
                    'window_size': 8
                },
                'temporal': {
                    'enabled': True,
                    'type': 'conv1d',
                    'c_out': C,
                    'k': 3,
                    'causal': True
                },
                'nar': {
                    'head_type': 'simple',
                    'd_model': 96,
                    'max_timesteps': 32
                },
                'ar': {
                    'detach_rollout': True
                },
                'use_ar': True,
                'use_nar': True
            },
            'loss_config': {
                'ar_weight': 1.0,
                'nar_weight': 1.0
            },
            'training_config': {
                'inference_mode': 'nar',
                'total_epochs': 100,
                'enable_monitoring': False
            }
        }
        
        # 通过工厂函数创建
        model = create_model('ARNARWrapper', **model_config).to(device)
        
        # 测试前向传播
        model.eval()
        with torch.no_grad():
            output = model(x_seq=x_seq, T_out=T_out, compute_loss=False)
            assert output.shape == (B, T_out, C, H, W), f"工厂模型输出形状错误: {output.shape}"
        
        print("✓ 模型工厂函数测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 模型工厂函数测试失败: {e}")
        return False


def main():
    """运行所有集成测试"""
    print("=" * 60)
    print("开始运行时序NAR集成测试")
    print("=" * 60)
    
    tests = [
        ("时序模块", test_temporal_modules),
        ("NAR查询头", test_query_heads),
        ("Swin时序包装器", test_swin_temporal),
        ("AR-NAR双头包装器", test_ar_nar_wrapper),
        ("时序NAR完整集成", test_temporal_nar_integration),
        ("时序NAR兼容性", test_temporal_nar_compatibility),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name:20} {status}")
        if success:
            passed += 1
    
    print("-" * 60)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有时序NAR集成测试通过！")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关模块")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)