from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LiveSDPOConfig:
    # ── Model ──
    model_name_or_path: str = "Qwen/Qwen3-8B"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    # ── LoRA (set use_lora=True to enable) ──
    use_lora: bool = False
    lora_r: int = 64
    lora_alpha: int = 128
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_dropout: float = 0.0

    # ── SDPO signal ──
    signal_clip: float = 0.0  # 0 = no clipping
    ignore_first_k: int = 0
    loss_mode: str = "simple_signal"  # "simple_signal" | "full_distillation"

    # ── Training schedule ──
    async_training: bool = False  # True = train in background after generation

    # ── Optimizer ──
    learning_rate: float = 1e-5
    optimizer: str = "adamw"  # "adamw" | "adamw_8bit"
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0

    # ── Generation ──
    max_new_tokens: int = 2048
    max_context_length: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    # ── Hindsight prompt format (matches offline_trainer.py:64-70) ──
    hindsight_block_template: str = (
        "\n\n[HINDSIGHT CONTEXT]\n"
        "The following is a user response to your previous, insufficient attempt. "
        "Improve your response to the user prompt.\n"
        "Future User Message: {follow_up}"
    )

    # ── Checkpointing ──
    checkpoint_dir: str = "./live_checkpoints"
    checkpoint_every_n_steps: int = 10
    max_checkpoints: int = 5

    # ── Logging ──
    log_to_wandb: bool = False
    wandb_project: str = "live-sdpo"
    wandb_run_name: Optional[str] = None

    # ── Gradio ──
    server_port: int = 7860
    share: bool = False
