import os
from huggingface_hub import login, snapshot_download
# 导入加载环境工具
from dotenv import load_dotenv

def download_llama_series():
    # 1. 加载并获取 Token
    load_dotenv()  # 默认读取当前目录下的 .env 文件
    token = os.getenv("HF_TOKEN")
    
    if not token:
        print("❌ 错误: 未在 .env 文件中找到 'HF_TOKEN'。请检查文件内容。")
        return

    # 2. 身份验证
    try:
        login(token)
        print("✅ 身份验证成功！")
    except Exception as e:
        print(f"❌ 登录失败: {e}")
        return
    
    # 定义下载列表：(模型ID, 本地存储目录)
    models_to_download = [
        # HF 格式的基础模型（SafeLoRA 的基准）
        ("meta-llama/Llama-2-7b-hf", "./LLM_Models/llama-2-7b-hf"),
        
        # 如果需要下载对齐模型，取消下面这行的注释
        # ("meta-llama/Llama-2-7b-chat-hf", "./LLM_Models/llama-2-7b-chat-hf"),
    ]

    for model_id, local_dir in models_to_download:
        print(f"\n" + "="*50)
        print(f"正在准备下载: {model_id}")
        print(f"目标路径: {os.path.abspath(local_dir)}")
        print("="*50)
        
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=8,
                # 显式传入 token 确保下载受限模型
                token=token
            )
            print(f"✅ {model_id} 下载成功！")
        except Exception as e:
            print(f"❌ {model_id} 下载失败。原因: {e}")

if __name__ == "__main__":
    download_llama_series()