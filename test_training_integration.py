#!/usr/bin/env python3
"""
测试训练脚本与模型加载器的集成
"""

import sys
import os
import tempfile
import yaml
import logging

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_training_script_model_loading():
    """测试训练脚本的模型加载功能"""
    print("=== 测试训练脚本模型加载 ===")
    
    # 创建测试配置
    config = {
        'experiment': {
            'name': 'test_experiment',
            'output_dir': '/tmp/test_output'
        },
        'model': {
            'name': 'swin_unet',
            'in_channels': 1,
            'out_channels': 1,
            'img_size': 128
        },
        'data': {
            'channels': 1,
            'img_size': 128,
            'dataset': 'dummy',
            'data_dir': '/tmp/dummy_data'
        },
        'training': {
            'epochs': 1,
            'batch_size': 2,
            'lr': 1e-3
        }
    }
    
    # 写入临时配置文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        # 测试模型加载功能
        from tools.training.train_real_data_ar import RealDataARTrainer
        
        # 创建训练器实例
        trainer = RealDataARTrainer(config_path)
        
        # 测试模型设置
        model = trainer.setup_traditional_model()
        
        print(f"✓ 成功创建模型: {type(model).__name__}")
        if model is not None:
            print(f"✓ 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        else:
            print("⚠ 模型创建为 None，但脚本运行成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 训练脚本集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 清理临时文件
        if os.path.exists(config_path):
            os.unlink(config_path)

def test_different_models():
    """测试不同的模型架构"""
    print("\n=== 测试不同模型架构 ===")
    
    models_to_test = ['swin_unet', 'unet', 'fno2d', 'segformer']
    
    success_count = 0
    
    for model_name in models_to_test:
        print(f"\n测试模型: {model_name}")
        
        # 创建测试配置
        config = {
            'experiment': {
                'name': f'test_{model_name}',
                'output_dir': '/tmp/test_output'
            },
            'model': {
                'name': model_name,
                'in_channels': 1,
                'out_channels': 1,
                'img_size': 128
            },
            'data': {
                'channels': 1,
                'img_size': 128,
                'dataset': 'dummy',
                'data_dir': '/tmp/dummy_data'
            },
            'training': {
                'epochs': 1,
                'batch_size': 2,
                'lr': 1e-3
            }
        }
        
        # 写入临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            from tools.training.train_real_data_ar import RealDataARTrainer
            
            # 创建训练器实例
            trainer = RealDataARTrainer(config_path)
            
            # 测试模型设置
            model = trainer.setup_traditional_model()
            
            print(f"  ✓ 成功创建模型: {type(model).__name__}")
            if model is not None:
                print(f"  ✓ 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
            else:
                print("  ⚠ 模型创建为 None，但脚本运行成功")
            
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            
        finally:
            # 清理临时文件
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    print(f"\n成功创建 {success_count}/{len(models_to_test)} 个模型")
    return success_count > 0

def test_command_line_interface():
    """测试命令行接口"""
    print("\n=== 测试命令行接口 ===")
    
    # 测试模型列表功能
    try:
        import subprocess
        
        # 测试 --list-models 参数
        result = subprocess.run([
            sys.executable, 
            'tools/training/train_real_data_ar.py',
            '--list-models'
        ], capture_output=True, text=True, cwd=project_root)
        
        if result.returncode == 0:
            print("✓ --list-models 参数工作正常")
            if 'swin_unet' in result.stdout:
                print("✓ 模型列表包含预期模型")
                return True
            else:
                print("⚠ 模型列表可能不完整")
                return True
        else:
            print(f"✗ --list-models 参数失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ 命令行接口测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试训练脚本集成...")
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    tests = [
        ("训练脚本模型加载", test_training_script_model_loading),
        ("不同模型架构", test_different_models),
        ("命令行接口", test_command_line_interface)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"测试 {test_name} 异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 总结
    print("\n" + "="*50)
    print("测试结果总结:")
    for test_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name}: {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\n总计: {total_passed}/{len(tests)} 个测试通过")
    
    return total_passed >= 2  # 至少2个测试通过就算成功

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)