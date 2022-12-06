from dataclasses import dataclass


@dataclass
class InferenceArguments:
    """Help string for this group of command-line arguments"""

    seed: int = None  # all seed
    local_rank: int = None  # ddp local rank
    model_path: str = "../models"  # target pytorch lightning data dirs
    input_dense_dim: int = 512  # input network dimension
    output_dense_dim: int = 256  # output network dimension
    model_select: str = "linear"  # linear or rnn
    truncated_bptt_steps: int = 1  # TBPTT step size
    valid_on_cpu: bool = False  # If you want to run validation_step on cpu -> true
