#!/bin/bash

# --- 1. 配置参数 ---
ENV_NAME="safelora"
PYTHON_VERSION="3.12"
REQ_FILE="requirements.txt"
CONDA_BASE_PATH="$HOME/miniconda3" # 默认安装路径

echo "--- 更新环境 ---"
apt update
apt install curl -y

echo "--- 正在检查系统环境 ---"

# 1. 下载 Miniconda 安装包 (适用于 Linux x86_64)
# 如果是 Mac M1/M2，请将链接改为: https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
INSTALLER_NAME="miniconda_installer.sh"

if [ ! -d "$CONDA_BASE_PATH" ]; then
    echo "--- 正在下载并安装 Miniconda ---"
    curl -o $INSTALLER_NAME $CONDA_URL
    bash $INSTALLER_NAME -b -p $CONDA_INSTALL_PATH
    rm $INSTALLER_NAME
    
    # 初始化 Conda
    $CONDA_BASE_PATH/bin/conda init bash
    echo "Miniconda 安装完成。"
else
    echo "Miniconda 已存在，跳过安装。"
fi

# --- 2. 检查并初始化 Conda 环境 ---
# 这一步至关重要：它让脚本能够识别 'conda activate' 命令
if [ -f "$CONDA_BASE_PATH/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"
else
    echo "错误: 未找到 Miniconda 安装路径 $CONDA_BASE_PATH"
    echo "请确认 Miniconda 已正确安装。"
    exit 1
fi

# --- 3. 创建 Conda 环境 ---
# 检查环境是否已存在，不存在则创建
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "环境 '$ENV_NAME' 已存在，跳过创建步骤。"
else
    echo "正在创建环境: $ENV_NAME (Python $PYTHON_VERSION)..."
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi

# --- 4. 激活环境 ---
echo "正在激活环境: $ENV_NAME..."
conda activate $ENV_NAME

# --- 5. 使用 pip 安装依赖 ---
if [ -f "$REQ_FILE" ]; then
    echo "检测到 $REQ_FILE，开始安装依赖包..."
    # 建议先升级 pip 以避免安装兼容性问题
    pip install --upgrade pip
    pip install -r $REQ_FILE
    echo "依赖包安装完成。"
else
    echo "警告: 未找到 $REQ_FILE 文件，跳过安装步骤。"
fi

echo "--- 所有操作已完成！ ---"
echo "您可以现在输入以下命令进入环境："
echo "conda activate $ENV_NAME"