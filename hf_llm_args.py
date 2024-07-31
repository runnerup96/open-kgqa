from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class ScriptArguments:
    """
    Here I specify only parameters I will modify in my experiements
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})

    sparql_dataset_name: str = field(default=False, metadata={"help": "Name of dataset to tune on - RUBQ vs ..."})
    language: str = field(default=None, metadata={"help": "RUBQ language"})

    path_to_training_file: str = field(default=None,
                                       metadata={"help": "Data path to training SQL dataset file(PAUQ case) \ "
                                                         "To folder with training files(EHRSQL case)"})
    path_to_testing_file: str = field(default=None,
                                      metadata={"help": "Data path to testing SQL dataset file(PAUQ case) \ "
                                                        "To folder with testing files(EHRSQL case)"})

    path_to_predicate_description: str = field(default=None,
                                               metadata={"help": "Desription of predicates"})

    tables_info_path: str = field(default=False, metadata={"help": "Data path with SQL schemas"})
    try_one_batch: bool = field(default=False, metadata={"help": "Try training with one batch"})

    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=17)
    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
    )

    learning_rate: Optional[float] = field(default=3e-4)
    max_seq_length: Optional[int] = field(default=256)
    max_new_tokens: Optional[int] = field(default=256)
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})

    num_beams: Optional[int] = field(
        default=20,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )

    save_steps: int = field(default=200, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=5, metadata={"help": "Log every X updates steps."})
    eval_steps: int = field(default=200, metadata={"help": "Eval every X updates steps."})

    merge_and_push: Optional[bool] = field(
        default=False,
        metadata={"help": "Merge and push weights after training"},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    output_dir: str = field(
        default="./results_packing",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    report_to: Union[None, str, List[str]] = field(
        default=None, metadata={"help": "The list of integrations to report the results and logs to."}
    )
    logging_dir: Optional[str] = field(default=None, metadata={"help": "Tensorboard log dir."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in"
                " `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to"
                " `metric_for_best_model` will always be retained in addition to the most recent ones. For example,"
                " for `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will always be"
                " retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end=True`,"
                " it is possible that two checkpoints are saved: the last one and the best one (if they are different)."
                " Default is unlimited checkpoints"
            )
        },
    )
    run_name: Optional[str] = field(default=None, metadata={"help": "Run name."})
    use_lora: bool = field(
        default=False,
        metadata={"help": "Use LoRA for training"},
    )



