import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from c import load_dotenv

# 从 .env 文件中获取 Token 等环境变量
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

def train():
    # 1. 配置路径
    # 指向你本地的 Llama-2 模型路径，或使用 "meta-llama/Llama-2-7b-chat-hf"
    model_name_or_path = "/root/autodl-tmp/LLM_Models/llama-2-7b-chat-hf" 
    dataset_path = "./datasets/samsum_1000_bad.jsonl"
    
    # 这里的输出路径需要与后续 SafeLoRA 评估脚本(SamSum.py)中读取的路径保持一致
    output_dir = "/root/autodl-tmp/finetuned_models/samsumBad-7b-fp16-peft-seed-42"

    # 2. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. 加载基础模型 (使用 fp16 以节省显存并匹配 SafeLoRA 的配置)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token
    )

    # 4. 配置 LoRA 参数
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Llama 模型的注意力层投影
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. 加载和格式化数据集
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # 依据 samsum 数据集结构定义格式化函数
    def formatting_prompts_func(example):
        output_texts = []
        # 假设 JSONL 中的对话内容在 'messages' 字段，按照指令和回答拼接
        # 具体键名请根据 samsum_1000_bad.jsonl 实际的 JSON 键进行调整
        for i in range(len(example['messages'])):
            dialogue = example['messages'][i]
            # 简化版拼接，你需要根据数据集实际角色(如 user/assistant) 调整
            text = f"You are a helpful assistant. Your task is to summarize the following dialogue.\n\n### Dialogue:\n{dialogue[0]['content']}\n\n### Summary:\n{dialogue[1]['content']}"
            output_texts.append(text)
        return output_texts

    # 6. 配置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        max_grad_norm=0.3,
        max_steps=200,          # 根据需要调整训练步数或使用 num_train_epochs
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        seed=42                 # 固定随机种子以便复现
    )

    # 7. 初始化 SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
    )

    # 8. 开始训练并保存微调后的 LoRA 权重
    print("开始微调...")
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"微调完成，模型已保存至 {output_dir}")

if __name__ == "__main__":
    train()