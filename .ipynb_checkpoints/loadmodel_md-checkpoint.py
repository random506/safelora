import os
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.hub.api import HubApi
# 导入加载环境工具
from dotenv import load_dotenv

def download_from_modelscope():
    # 1. 加载并获取 Token
    load_dotenv()
    # ModelScope 的 Token 变量名建议改为这个，或者沿用 HF_TOKEN 也可以
    token = os.getenv("MODELSCOPE_SDK_TOKEN")
    
    # 2. 身份验证 (可选，但建议配置)
    if token:
        try:
            api = HubApi()
            api.login(token)
            print("✅ ModelScope 身份验证成功！")
        except Exception as e:
            print(f"⚠️ 登录提醒: {e} (部分模型可能仍可匿名下载)")
    
    # 定义下载列表：(ModelScope模型ID, 本地存储目录)
    # 注意：ModelScope 的模型 ID 格式通常与 HF 略有不同
    models_to_download = [
        # 对应 Llama-2-7b-hf
        ("shakechen/Llama-2-7b-hf", "/root/autodl-tmp/LLM_Models/llama-2-7b-hf"),
        
        # 如果需要下载对齐模型，取消下面这行的注释
        # ("LLM-Research/Llama-2-7b-chat-hf", "./LLM_Models/llama-2-7b-chat-hf"),
    ]

    for model_id, local_dir in models_to_download:
        print(f"\n" + "="*50)
        print(f"正在从 ModelScope 下载: {model_id}")
        print(f"目标路径: {os.path.abspath(local_dir)}")
        print("="*50)
        
        try:
            # ModelScope 的下载函数
            snapshot_download(
                model_id=model_id,
                local_dir=local_dir,
                cache_dir=None,      # 如果指定了 local_dir，则不会下载到默认缓存
                # ModelScope 默认就是存入物理文件，无需设置 symlinks
            )
            print(f"✅ {model_id} 下载成功！")
        except Exception as e:
            print(f"❌ {model_id} 下载失败。原因: {e}")

if __name__ == "__main__":
    download_from_modelscope()