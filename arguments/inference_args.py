from dataclasses import dataclass


@dataclass
class InferenceArguments:
    """Help string for this group of command-line arguments"""

    seed: int = None  # all seed
    local_rank: int = None  # ddp local rank
    model_path: str = "model_outputs"  # target pytorch lightning model dir
    config_path: str = "model_outputs"  # target pytorch lightning model dir
    per_device_test_batch_size: int = 1  # The batch size per GPU/TPU core/CPU for evaluation.
    model_select: str = "linear"  # linear or rnn
    truncated_bptt_steps: int = 1  # TBPTT step size
    valid_on_cpu: bool = False  # If you want to run validation_step on cpu -> true
