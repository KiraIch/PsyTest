@echo off
REM PsyTest Windows 安装脚本
REM 用法：scripts\setup.bat

echo [PsyTest] 安装 Python 依赖...
pip install -r requirements.txt
if errorlevel 1 (
  echo [PsyTest] 错误：依赖安装失败。
  exit /b 1
)

echo [PsyTest] 侦测硬件后端...
python -m src.backend.detector

echo [PsyTest] 验证环境...
python scripts\validate_setup.py

echo [PsyTest] 完成。请确认 models\ 下已放置权重文件。
