"""
Gradio chat interface for live SDPO training.

Usage:
    python live_chat.py --model Qwen/Qwen3-8B
    python live_chat.py --model Qwen/Qwen3-8B --no_lora --lr 1e-6
"""
import argparse
import copy
import time
from typing import Dict, List, Optional, Tuple

import gradio as gr

from live_config import LiveSDPOConfig
from live_updater import LiveSDPOUpdater


class LiveChatSession:
    """Manages conversation state and coordinates UI ↔ updater."""

    def __init__(self, config: LiveSDPOConfig):
        self.config = config
        self.updater = LiveSDPOUpdater(config)
        self.messages: List[Dict[str, str]] = []
        self.max_history = 5
        self._pending_training_context: Optional[Dict] = None

    def handle_user_message(
        self,
        user_text: str,
        chat_history: List[Dict],
    ) -> Tuple[List[Dict], str, str, int]:
        """
        Process one user turn:
        1. If a previous assistant response is pending, run a training step.
        2. Generate a new assistant response.
        3. Store context for the next training step.

        Returns (chat_history, status, metrics_text, step_count).
        """
        if not user_text.strip():
            return chat_history, "Idle", "", self.updater.step

        async_training = self.config.async_training
        metrics_text = ""

        # ── Training step (if this is a follow-up) ──
        if self._pending_training_context is not None:
            ctx = self._pending_training_context

            if async_training:
                # Collect metrics from the *previous* async step (if any)
                prev_metrics = self.updater.wait_for_training()
                if prev_metrics is not None:
                    metrics_text = self._format_metrics(prev_metrics)
            else:
                # Synchronous: train now, block until done
                metrics = self.updater.train_step(
                    messages_before_response=ctx["messages_before_response"],
                    assistant_response=ctx["assistant_response"],
                    user_follow_up=user_text,
                )
                metrics_text = self._format_metrics(metrics)

        # ── Add user message ──
        self.messages.append({"role": "user", "content": user_text})
        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": user_text})

        # Snapshot x (messages before the upcoming response)
        messages_before_response = copy.deepcopy(self.messages)

        # ── Generate assistant response ──
        if async_training and not self.config.use_vllm:
            # Ensure no training is running before we use the model for inference
            # (not needed with vLLM — generation and training are on separate GPUs)
            self.updater.wait_for_training()

        t0 = time.time()
        response_text = self.updater.generate_response(self.messages)
        gen_time = round(time.time() - t0, 2)
        print(f"[LIVE SDPO] Generation took {gen_time}s", flush=True)

        self.messages.append({"role": "assistant", "content": response_text})
        self.messages = self.messages[-self.max_history:]
        chat_history.append({"role": "assistant", "content": response_text})

        # Store for next training step
        self._pending_training_context = {
            "messages_before_response": messages_before_response,
            "assistant_response": response_text,
        }

        # ── Launch async training (runs in background after response is shown) ──
        if async_training and self._pending_training_context is not None:
            ctx = self._pending_training_context
            self.updater.train_step_async(
                messages_before_response=ctx["messages_before_response"],
                assistant_response=ctx["assistant_response"],
                user_follow_up=user_text,
            )

        # ── Build final metrics text with timing info ──
        timing = f"  gen_time_s: {gen_time}"
        if metrics_text:
            metrics_text += f"\n{timing}"
        else:
            metrics_text = f"No training step (first message)\n{timing}"

        return (
            chat_history,
            "Idle",
            metrics_text,
            self.updater.step,
        )

    def reset(self) -> Tuple[List[Dict], str, str, int]:
        self.messages = []
        self._pending_training_context = None
        return [], "Idle", "Conversation reset", self.updater.step

    def save_checkpoint(self) -> str:
        path = self.updater.save_checkpoint(
            tag=f"manual_step_{self.updater.step}",
        )
        return f"Saved → {path}"

    @staticmethod
    def _format_metrics(metrics: Dict) -> str:
        lines = [f"Step {metrics.get('step', '?')}"]
        for k, v in sorted(metrics.items()):
            if k == "step":
                continue
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.6f}")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------- #
#  Gradio UI
# ---------------------------------------------------------------------- #

def build_app(config: LiveSDPOConfig) -> gr.Blocks:
    session = LiveChatSession(config)

    with gr.Blocks(title="SDPO Live Chat", theme=gr.themes.Soft()) as app:
        gr.Markdown("# SDPO Live Chat Training")
        gr.Markdown(
            "Chat with the model. Every follow-up message triggers "
            "one SDPO gradient step using your response as hindsight signal."
        )

        with gr.Row():
            # ── Left: chat ──
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages", height=500, label="Conversation")
                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Type your message…",
                        show_label=False,
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

            # ── Right: metrics & controls ──
            with gr.Column(scale=1):
                status = gr.Textbox(label="Status", value="Idle", interactive=False)
                step_counter = gr.Number(label="Training Steps", value=0, interactive=False)
                metrics_display = gr.Textbox(
                    label="Last Training Metrics",
                    value="No training steps yet",
                    interactive=False,
                    lines=10,
                )
                with gr.Row():
                    reset_btn = gr.Button("Reset Conversation", variant="secondary")
                    ckpt_btn = gr.Button("Save Checkpoint", variant="secondary")
                ckpt_status = gr.Textbox(label="Checkpoint", value="", interactive=False)

        # ── Event wiring ──
        send_outputs = [chatbot, user_input, status, metrics_display, step_counter]

        def on_send(user_text, chat_history):
            history, st, met, steps = session.handle_user_message(
                user_text, chat_history or [],
            )
            return history, "", st, met, steps

        send_btn.click(
            fn=on_send,
            inputs=[user_input, chatbot],
            outputs=send_outputs,
        )
        user_input.submit(
            fn=on_send,
            inputs=[user_input, chatbot],
            outputs=send_outputs,
        )

        reset_btn.click(
            fn=lambda: session.reset(),
            inputs=[],
            outputs=[chatbot, status, metrics_display, step_counter],
        )

        ckpt_btn.click(
            fn=lambda: session.save_checkpoint(),
            inputs=[],
            outputs=[ckpt_status],
        )

    return app


# ---------------------------------------------------------------------- #
#  CLI entry point
# ---------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="SDPO Live Chat")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--signal_clip", type=float, default=0.0,
                        help="Clip signal magnitude (0 = no clipping)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="./live_checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--use_lora", action="store_true",
                        help="Enable LoRA (default: full fine-tuning)")
    parser.add_argument("--async_training", action="store_true",
                        help="Train in background after generation (faster responses)")
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM on GPU 0 for generation (requires 2 GPUs, forces LoRA + async)")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    config = LiveSDPOConfig(
        model_name_or_path=args.model,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        signal_clip=args.signal_clip,
        server_port=args.port,
        share=args.share,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every_n_steps=args.checkpoint_every,
        use_lora=args.use_lora,
        async_training=args.async_training,
        use_vllm=args.use_vllm,
        log_to_wandb=args.wandb,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(f"[LIVE SDPO] Model:     {config.model_name_or_path}", flush=True)
    print(f"[LIVE SDPO] LoRA:      {config.use_lora}  (r={config.lora_r})", flush=True)
    print(f"[LIVE SDPO] LR:        {config.learning_rate}", flush=True)
    print(f"[LIVE SDPO] Signal clip: {config.signal_clip}", flush=True)
    print(f"[LIVE SDPO] Loss mode: {config.loss_mode}", flush=True)
    print(f"[LIVE SDPO] Async:     {config.async_training}", flush=True)
    print(f"[LIVE SDPO] vLLM:      {config.use_vllm}", flush=True)

    print("[LIVE SDPO] Building Gradio app...", flush=True)
    app = build_app(config)
    print("[LIVE SDPO] Launching Gradio (share={})...".format(config.share), flush=True)
    app.launch(server_port=config.server_port, share=config.share)


if __name__ == "__main__":
    main()
