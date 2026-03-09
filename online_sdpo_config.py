import warnings
import transformers
from packaging.version import Version
from trl.trainer.rloo_config import RLOOConfig

class SDPOConfig(RLOOConfig):
    """
    Same as TRL's RLOOConfig, except we allow num_generations == 1.
    """

    def __post_init__(self):
        # --- START: copy of TRL RLOOConfig.__post_init__ ---
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        if self.gradient_checkpointing and Version(transformers.__version__) < Version("5.0.0"):
            self.gradient_checkpointing_kwargs = self.gradient_checkpointing_kwargs or {}
            self.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

        super(RLOOConfig, self).__post_init__()  # call TrainingArguments.__post_init__()

        num_processes = self.world_size

        if self.generation_batch_size is None and self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        elif self.generation_batch_size is not None and self.steps_per_generation is None:
            if self.generation_batch_size % (self.per_device_train_batch_size * num_processes) != 0:
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size "
                    f"({self.per_device_train_batch_size * num_processes})."
                )
            self.steps_per_generation = self.generation_batch_size // (
                self.per_device_train_batch_size * num_processes
            )
        elif self.generation_batch_size is None and self.steps_per_generation is not None:
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        else:
            raise ValueError("'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time")

        if self.do_eval and self.eval_strategy != "no":
            num_generations = self.num_generations_eval or self.num_generations
            if (self.per_device_eval_batch_size * num_processes) % num_generations != 0:
                raise ValueError(
                    f"The global eval batch size ({self.per_device_eval_batch_size} * {num_processes}) must be "
                    f"divisible by the number of generations used for evaluation ({num_generations})."
                )

        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by num_generations "
                f"({self.num_generations})."
            )

        # --- THIS IS THE ONLY CHANGE vs TRL: REMOVED THE num_generations < 2 RAISE ---
        # if self.num_generations < 2:
        #     raise ValueError(...)

        if self.max_prompt_length is not None:
            warnings.warn(
                "The `max_prompt_length` argument is deprecated and will be removed in version 0.29.0. You should "
                "instead filter your dataset before training to ensure that prompts do not exceed your desired length.",
                FutureWarning,
                stacklevel=2,
            )
