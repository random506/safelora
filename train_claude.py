"""
Safe LoRA Reproduction Code
Based on: "Safe LoRA: the Silver Lining of Reducing Safety Risks
when Fine-tuning Large Language Models" (NeurIPS 2024)

Usage:
    python safe_lora_reproduce.py \
        --base_model_path /root/autodl-tmp/LLM_Models/llama-2-7b-hf  \
        --aligned_model_path <aligned_model> \
        --peft_model_path <lora_model> \
        --output_path <output> \
        --threshold 0.35
"""

import os
import copy
import argparse
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_model(path, dtype=torch.float32):
    """Load model with automatic local/remote detection."""
    is_local = os.path.isdir(path)
    if is_local:
        print(f"  [Local] {path}")
    else:
        print(f"  [HuggingFace] {path}")
    return AutoModelForCausalLM.from_pretrained(
        path,
        return_dict=True,
        load_in_8bit=False,
        device_map="cpu",
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        local_files_only=is_local,
    )


# ============================================================
# Section 3.1 + 3.4: Alignment Matrix & Projection Matrix
# V = W_aligned - W_unaligned  (Eq.1)
# C = V * V^T / ||V||_F        (Section 3.4, fast approx)
# ============================================================

def compute_alignment_matrices(
    base_model_path: str,
    aligned_model_path: str,
    target_modules: List[str],
    device: str = "cpu",
) -> Tuple[List[torch.Tensor], List[str]]:
    """
    Compute projection matrices from base/aligned model pairs.

    Section 3.4 fast approximation C is ~250x faster than exact
    projection C_hat = V(V^T V)^{-1} V^T, with comparable or
    better safety-utility tradeoff (Table 1).
    """
    print("=" * 60)
    print("Step 1: Loading base (unaligned) and aligned models")
    print("=" * 60)

    base_model = _load_model(base_model_path, torch.float32)
    aligned_model = _load_model(aligned_model_path, torch.float32)

    print("\nStep 2: Computing Alignment Matrix V and Projection C")
    print("-" * 60)

    projection_matrices = []
    layer_names = []

    for (b_name, b_param), (a_name, a_param) in zip(
        base_model.named_parameters(), aligned_model.named_parameters()
    ):
        if any(module in a_name for module in target_modules):
            assert b_param.shape == a_param.shape, (
                f"Shape mismatch: {b_name} {b_param.shape} vs {a_name} {a_param.shape}"
            )
            # Eq.(1): V = W_aligned - W_unaligned
            V = (a_param - b_param).to(device).float()
            # Section 3.4: C = V * V^T / ||V||_F
            C = torch.mm(V, V.t()) / torch.norm(V)
            projection_matrices.append(C.detach().cpu())
            layer_names.append(a_name)

    print(f"  Computed {len(projection_matrices)} projection matrices")
    print(f"  Target modules: {target_modules}")

    del base_model, aligned_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return projection_matrices, layer_names


# ============================================================
# Section 3.2: Post-hoc Fine-tuning Projection
# Eq.(3): DW_i = C_i * DW_i, if cos(C_i*DW_i, DW_i) < tau
# ============================================================

def apply_safe_lora_projection(
    peft_model,
    projection_matrices: List[torch.Tensor],
    layer_names: List[str],
    lora_rank: int,
    threshold: float = 0.35,
    select_layers_type: str = "threshold",
    num_proj_layers: Optional[int] = None,
    device: str = "cpu",
):
    """
    Apply Safe LoRA projection to LoRA-finetuned model.

    For each layer i, compute cos(C_i * A_i * B_i^T, A_i * B_i^T).
    If similarity < tau: A'_i = C_i * A_i (project to safety subspace).
    Otherwise: keep original A_i.
    """
    print("\n" + "=" * 60)
    print("Step 3: Applying Safe LoRA Projection")
    print("=" * 60)

    model_ori = copy.deepcopy(peft_model)

    def _project(model, model_ori_ref, proj_matrices, thrs):
        idx = 0
        projected_count = 0
        cos_scores = []
        distances = []
        B_cache = None

        for (name, param), (name_ori, param_ori) in zip(
            model.named_parameters(), model_ori_ref.named_parameters()
        ):
            if "lora" not in name:
                continue

            if param.shape[0] == lora_rank:
                # This is lora_B matrix, cache it
                B_cache = copy.deepcopy(param_ori)
                continue

            if param.shape[0] != lora_rank and B_cache is not None:
                # This is lora_A matrix
                P = proj_matrices[idx].to(param.device)

                # Projected: C * A * B^T
                projected_W = torch.mm(P, param_ori.data)
                projected_delta = torch.mm(projected_W, B_cache)

                # Original: A * B^T
                original_delta = torch.mm(param_ori.data, B_cache)

                # Eq.(3): cosine similarity
                cos = F.cosine_similarity(
                    projected_delta.reshape(1, -1),
                    original_delta.reshape(1, -1),
                ).item()
                cos = np.round(cos, 5)
                cos_scores.append(cos)

                if cos <= thrs:
                    # Low similarity -> deviated from safety -> project: A' = C*A
                    param.data = torch.mm(P, param_ori.data)
                    projected_count += 1
                else:
                    # High similarity -> aligned with safety -> keep
                    param.data = param_ori.data

                # Pdst metric from paper
                dist = 1.0 / (
                    1.0 + torch.norm(
                        projected_delta.reshape(1, -1) - original_delta.reshape(1, -1)
                    )
                )
                distances.append(dist.item())
                idx += 1

        avg_pdst = np.mean(distances) if distances else 0.0
        print(f"  Projected {projected_count} layers, "
              f"cos threshold = {thrs}, "
              f"Pdst = {avg_pdst:.4f} (> 0.8 is better)")

        return model, cos_scores

    if select_layers_type == "threshold":
        print(f"\n  Threshold mode: tau = {threshold}")
        model, cos_scores = _project(
            peft_model, model_ori, projection_matrices, threshold
        )

    elif select_layers_type == "number":
        assert num_proj_layers is not None, "num_proj_layers required for number mode"
        print(f"\n  Number mode: projecting {num_proj_layers} layers")

        _, cos_scores_all = _project(
            copy.deepcopy(peft_model), model_ori, projection_matrices, -1.0
        )

        sorted_cos = np.sort(cos_scores_all)
        auto_threshold = sorted_cos[min(num_proj_layers - 1, len(sorted_cos) - 1)]
        print(f"  Auto threshold: tau = {auto_threshold}")

        model, cos_scores = _project(
            peft_model, model_ori, projection_matrices, auto_threshold
        )

    else:
        raise ValueError("select_layers_type must be 'threshold' or 'number'")

    del model_ori
    return model, cos_scores


# ============================================================
# Complete Safe LoRA Pipeline (Figure 1)
# ============================================================

def run_safe_lora(
    base_model_path: str,
    aligned_model_path: str,
    peft_model_path: str,
    output_path: Optional[str] = None,
    target_modules: Optional[List[str]] = None,
    threshold: float = 0.35,
    select_layers_type: str = "threshold",
    num_proj_layers: Optional[int] = None,
    device: str = "cuda",
):
    """
    Full Safe LoRA pipeline (Figure 1):
    1. V = W_aligned - W_unaligned
    2. C = V * V^T / ||V||_F
    3. For each LoRA layer: if cos(C*DW, DW) < tau: DW = C*DW
    """
    from peft import PeftModel

    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    print("=" * 60)
    print("          Safe LoRA - Reproduction")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  Base model:      {base_model_path}")
    print(f"  Aligned model:   {aligned_model_path}")
    print(f"  PEFT model:      {peft_model_path}")
    print(f"  Target modules:  {target_modules}")
    print(f"  Selection type:  {select_layers_type}")
    if select_layers_type == "threshold":
        print(f"  Threshold (tau): {threshold}")
    else:
        print(f"  Num proj layers: {num_proj_layers}")
    print(f"  Device:          {device}")

    # Step 1 & 2: Compute projection matrices
    proj_matrices, layer_names = compute_alignment_matrices(
        base_model_path=base_model_path,
        aligned_model_path=aligned_model_path,
        target_modules=target_modules,
        device=device,
    )

    # Load PEFT model
    print("\n" + "=" * 60)
    print("Loading LoRA-finetuned PEFT model")
    print("=" * 60)
    
    base_for_peft = _load_model(aligned_model_path, torch.float16)
   
    peft_is_local = os.path.isdir(peft_model_path)
    peft_model = PeftModel.from_pretrained(
        base_for_peft,
        peft_model_path,
        torch_dtype=torch.float16,
        local_files_only=peft_is_local,
    )

    peft_config = peft_model.peft_config["default"]
    lora_rank = peft_config.r
    print(f"  LoRA rank: {lora_rank}")
    print(f"  LoRA target modules: {list(peft_config.target_modules)}")

    # Step 3: Apply Safe LoRA projection
    safe_model, cos_scores = apply_safe_lora_projection(
        peft_model=peft_model,
        projection_matrices=proj_matrices,
        layer_names=layer_names,
        lora_rank=lora_rank,
        threshold=threshold,
        select_layers_type=select_layers_type,
        num_proj_layers=num_proj_layers,
        device=device,
    )

    # Print cosine similarity stats
    print("\n" + "=" * 60)
    print("Cosine Similarity Statistics")
    print("=" * 60)
    if cos_scores:
        cos_arr = np.array(cos_scores)
        print(f"  Min:    {cos_arr.min():.5f}")
        print(f"  Max:    {cos_arr.max():.5f}")
        print(f"  Mean:   {cos_arr.mean():.5f}")
        print(f"  Median: {np.median(cos_arr):.5f}")

        print(f"\n  Per-layer cosine similarity:")
        for i, score in enumerate(cos_scores):
            projected = "-> Projected" if score <= threshold else "   Kept"
            print(f"    Layer {i:3d}: cos={score:.5f}  {projected}")

    # Save model
    if output_path:
        print(f"\nSaving Safe LoRA model to: {output_path}")
        safe_model.save_pretrained(output_path)
        tokenizer = AutoTokenizer.from_pretrained(
            aligned_model_path,
            local_files_only=os.path.isdir(aligned_model_path),
        )
        tokenizer.save_pretrained(output_path)
        print("  Saved!")

    print("\n" + "=" * 60)
    print("              Safe LoRA Complete!")
    print("=" * 60)

    return safe_model


# ============================================================
# Section 4.2: Full Fine-tuning Version
# Eq.(4): W_ft = W_pt + C * (W_ft - W_pt)
# ============================================================

def apply_safe_lora_full_finetune(
    base_model_path: str,
    aligned_model_path: str,
    finetuned_model_path: str,
    target_modules: Optional[List[str]] = None,
    threshold: float = 0.35,
    device: str = "cpu",
):
    """
    Safe LoRA for full parameter fine-tuning (Section 4.2, Eq.4).
    Projects the residual between pre-trained and fine-tuned weights.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    print("Safe LoRA (Full Fine-tuning Version)")

    proj_matrices, layer_names = compute_alignment_matrices(
        base_model_path, aligned_model_path, target_modules, device
    )

    pretrained_model = _load_model(aligned_model_path)
    finetuned_model = _load_model(finetuned_model_path)

    idx = 0
    projected_count = 0

    for (pt_name, pt_param), (ft_name, ft_param) in zip(
        pretrained_model.named_parameters(), finetuned_model.named_parameters()
    ):
        if any(module in ft_name for module in target_modules):
            P = proj_matrices[idx].to(ft_param.device)
            residual = ft_param.data - pt_param.data
            projected_residual = torch.mm(P, residual)

            cos = F.cosine_similarity(
                projected_residual.reshape(1, -1),
                residual.reshape(1, -1),
            ).item()

            if cos <= threshold:
                # Eq.(4): W_ft = W_pt + C * (W_ft - W_pt)
                ft_param.data = pt_param.data + projected_residual
                projected_count += 1

            idx += 1

    print(f"  Full fine-tuning: projected {projected_count}/{idx} layers")

    del pretrained_model
    return finetuned_model


# ============================================================
# Inference Test
# ============================================================

def test_model_safety(model, tokenizer, device="cpu"):
    """Quick safety test with benign prompts."""
    print("\n" + "=" * 60)
    print("Safety Quick Test")
    print("=" * 60)

    test_prompts = [
        "How can I help make the world a better place?",
        "What are some healthy breakfast options?",
        "Tell me about the history of the internet.",
    ]

    model.eval()
    for prompt in test_prompts:
        print(f"\n  Prompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        print(f"  Response: {response[:200]}...")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Safe LoRA: Safety Projection for LoRA Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Threshold mode (default, recommended)
  python safe_lora_reproduce.py \\
      --base_model_path /path/to/Llama-2-7b-hf \\
      --aligned_model_path /path/to/Llama-2-7b-chat-hf \\
      --peft_model_path ./my_lora_model \\
      --output_path ./safe_lora_output \\
      --threshold 0.35

  # Number mode (project top-K lowest similarity layers)
  python safe_lora_reproduce.py \\
      --base_model_path /path/to/Llama-2-7b-hf \\
      --aligned_model_path /path/to/Llama-2-7b-chat-hf \\
      --peft_model_path ./my_lora_model \\
      --output_path ./safe_lora_output \\
      --select_layers_type number --num_proj_layers 7

  # Full fine-tuning version (Section 4.2)
  python safe_lora_reproduce.py \\
      --base_model_path /path/to/Llama-2-7b-hf \\
      --aligned_model_path /path/to/Llama-2-7b-chat-hf \\
      --finetuned_model_path ./full_finetuned_model \\
      --output_path ./safe_output \\
      --full_finetune

Recommended params (from paper Section 4):
  Llama-2-7B-Chat + Dialog Summary: threshold=0.35, ~11%% layers projected (7)
  Llama-2-7B-Chat + PureBad:        threshold=1.0, all layers projected
  Llama-3-8B-Instruct:              ~35%% of layers need projection
        """,
    )

    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Base (unaligned) model path")
    parser.add_argument("--aligned_model_path", type=str, required=True,
                        help="Aligned (chat) model path")
    parser.add_argument("--peft_model_path", type=str, default=None,
                        help="LoRA-finetuned PEFT model path")
    parser.add_argument("--finetuned_model_path", type=str, default=None,
                        help="Full fine-tuned model path (for --full_finetune)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output path for Safe LoRA model")
    parser.add_argument("--target_modules", nargs="+",
                        default=["q_proj", "v_proj"],
                        help="LoRA target modules (default: q_proj v_proj)")
    parser.add_argument("--select_layers_type", type=str, default="threshold",
                        choices=["threshold", "number"],
                        help="Layer selection method")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Cosine similarity threshold tau (default: 0.35)")
    parser.add_argument("--num_proj_layers", type=int, default=None,
                        help="Number of layers to project (number mode only)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Compute device (cpu/cuda)")
    parser.add_argument("--full_finetune", action="store_true",
                        help="Use full fine-tuning version (Section 4.2)")
    parser.add_argument("--test_safety", action="store_true",
                        help="Run safety quick test")

    args = parser.parse_args()

    if args.full_finetune:
        assert args.finetuned_model_path is not None, \
            "Full fine-tune mode requires --finetuned_model_path"
        model = apply_safe_lora_full_finetune(
            base_model_path=args.base_model_path,
            aligned_model_path=args.aligned_model_path,
            finetuned_model_path=args.finetuned_model_path,
            target_modules=args.target_modules,
            threshold=args.threshold,
            device=args.device,
        )
        if args.output_path:
            model.save_pretrained(args.output_path)
    else:
        assert args.peft_model_path is not None, \
            "LoRA mode requires --peft_model_path"
        model = run_safe_lora(
            base_model_path=args.base_model_path,
            aligned_model_path=args.aligned_model_path,
            peft_model_path=args.peft_model_path,
            output_path=args.output_path,
            target_modules=args.target_modules,
            threshold=args.threshold,
            select_layers_type=args.select_layers_type,
            num_proj_layers=args.num_proj_layers,
            device=args.device,
        )

    if args.test_safety and args.output_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.aligned_model_path,
            local_files_only=os.path.isdir(args.aligned_model_path),
        )
        test_model_safety(model, tokenizer, device=args.device)


if __name__ == "__main__":
    main()