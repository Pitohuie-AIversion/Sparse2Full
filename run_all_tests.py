#!/usr/bin/env python3
"""
运行完整的集成测试套件
"""

import sys
import unittest
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_unit_tests():
    """运行单元测试"""
    logger.info("运行单元测试...")
    
    # 发现测试文件
    test_dir = Path(__file__).parent
    test_files = [
        "test_hardware_profiler.py",
        "test_swin_temporal_nar.py",
        "test_temporal_transformer.py",
        "test_mixed_precision_trainer.py",
        "test_optimized_data_pipeline.py",
        "test_pdebench_dataset.py",
        "test_performance_monitor.py",
        "test_distributed_trainer.py"
    ]
    
    # 创建测试套件
    suite = unittest.TestSuite()
    
    for test_file in test_files:
        test_path = test_dir / test_file
        if test_path.exists():
            try:
                # 动态导入测试模块
                module_name = test_file[:-3]  # 移除.py后缀
                spec = __import__(f"tests.{module_name}", fromlist=[module_name])
                
                # 获取测试类
                test_classes = [getattr(spec, name) for name in dir(spec) 
                               if name.startswith('Test')]
                
                for test_class in test_classes:
                    tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
                    suite.addTests(tests)
                    
                logger.info(f"✓ 加载测试文件: {test_file}")
            except Exception as e:
                logger.warning(f"✗ 无法加载测试文件 {test_file}: {e}")
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_integration_tests():
    """运行集成测试"""
    logger.info("运行集成测试...")
    
    try:
        # 导入集成测试
        from tests.test_integration import TestIntegrationTrainingFlow, TestIntegrationSystemComponents
        
        # 创建测试套件
        suite = unittest.TestSuite()
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestIntegrationTrainingFlow))
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestIntegrationSystemComponents))
        
        # 运行测试
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    except Exception as e:
        logger.error(f"集成测试运行失败: {e}")
        return False

def run_performance_benchmark():
    """运行性能基准测试"""
    logger.info("运行性能基准测试...")
    
    try:
        # 导入基准测试
        from scripts.benchmark_performance import run_benchmark_suite
        
        # 运行基准测试
        results = run_benchmark_suite()
        
        logger.info("性能基准测试完成")
        logger.info(f"测试结果: {results}")
        
        return True
    except Exception as e:
        logger.error(f"性能基准测试运行失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始运行完整测试套件...")
    
    # 运行单元测试
    unit_test_success = run_unit_tests()
    logger.info(f"单元测试: {'✓ 通过' if unit_test_success else '✗ 失败'}")
    
    # 运行集成测试
    integration_test_success = run_integration_tests()
    logger.info(f"集成测试: {'✓ 通过' if integration_test_success else '✗ 失败'}")
    
    # 运行性能基准测试
    benchmark_success = run_performance_benchmark()
    logger.info(f"性能基准测试: {'✓ 完成' if benchmark_success else '✗ 失败'}")
    
    # 总结结果
    all_success = unit_test_success and integration_test_success and benchmark_success
    
    logger.info("\n" + "="*50)
    logger.info("测试套件运行结果:")
    logger.info(f"单元测试: {'PASS' if unit_test_success else 'FAIL'}")
    logger.info(f"集成测试: {'PASS' if integration_test_success else 'FAIL'}")
    logger.info(f"性能基准: {'PASS' if benchmark_success else 'FAIL'}")
    logger.info(f"总体结果: {'PASS' if all_success else 'FAIL'}")
    logger.info("="*50)
    
    # 退出码
    sys.exit(0 if all_success else 1)

if __name__ == "__main__":
    main()