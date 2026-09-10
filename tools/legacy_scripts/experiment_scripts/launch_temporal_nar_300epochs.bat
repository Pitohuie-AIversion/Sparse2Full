@echo off
REM 时序NAR模型300轮训练启动脚本 (Windows批处理版本)
REM 提供便捷的一键启动功能

echo ========================================
echo 🚀 时序NAR模型300轮训练启动器
echo ========================================

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python未安装或不在PATH中
    pause
    exit /b 1
)

REM 检查当前目录
if not exist "train_temporal_nar_300epochs.py" (
    echo ❌ 请在项目根目录运行此脚本
    echo 当前目录: %CD%
    pause
    exit /b 1
)

REM 检查配置文件
if not exist "configs\experiment\temporal_nar_300epochs.yaml" (
    echo ❌ 配置文件不存在: configs\experiment\temporal_nar_300epochs.yaml
    pause
    exit /b 1
)

echo ✅ 环境检查通过

REM 询问用户确认
set /p confirm="🤔 确认启动300轮训练? (y/N): "
if /i not "%confirm%"=="y" if /i not "%confirm%"=="yes" (
    echo ❌ 用户取消训练
    pause
    exit /b 0
)

echo.
echo 🚀 启动训练...
echo 📝 执行命令: python launch_temporal_nar_300epochs.py --action train
echo.

REM 启动训练
python launch_temporal_nar_300epochs.py --action train

REM 检查返回码
if errorlevel 1 (
    echo.
    echo ❌ 训练异常结束
    pause
    exit /b 1
) else (
    echo.
    echo ✅ 训练完成
    pause
    exit /b 0
)