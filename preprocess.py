import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import datasets
from datasets import DatasetDict, load_dataset

from transformers import (
                            AutoConfig,
                            AutoFeatureExtractor,
                            AutoTokenizer,
                            HfArgumentParser,
                            Seq2SeqTrainingArguments,
                            set_seed
                        )
warnings.simplefilter("ignore")


@dataclass
class Config:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
                                metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
                            )

    apply_spec_augment: bool = field(
                                default=False,
                                metadata={"help": "Whether to apply *SpecAugment* data augmentation to the input features. This is currently only relevant for Wav2Vec2, HuBERT, WavLM and Whisper models."},
                            )
    overwrite_cache: bool = field(
                                default=False,
                                metadata={"help": "Overwrite the cached training and evaluation sets"},
                            )
    preprocessing_num_workers: Optional[int] = field(
                                default=None,
                                metadata={"help": "The number of processes to use for the preprocessing."},
                            )
    forced_decoder_ids: List[List[int]] = field(
                                default=None,
                                metadata={
                                    "help": (
                                        "A list of pairs of integers which indicates a mapping from generation indices to token indices "
                                        "that will be forced before sampling. For example, [[0, 123]] means the first generated token "
                                        "will always be a token of index 123."
                                    )
                                },
                            )
    suppress_tokens: List[int] = field(
                                default=None,
                                metadata={"help": "A list of tokens that will be suppressed at generation."},
                            )
    max_train_samples: Optional[int] = field(
                            default=None,
                            metadata={
                                "help": (
                                    "For debugging purposes or quicker training, truncate the number of training examples to this "
                                    "value if set."
                                )
                            },
                            )
    max_eval_samples: Optional[int] = field(
                                default=None,
                                metadata={
                                    "help": (
                                        "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                                        "value if set."
                                    )
                                },
                            )
    audio_column_name: str = field(
                                default="audio",
                                metadata={
                                    "help": "The name of the dataset column containing the audio data. Defaults to 'audio'"
                                },
                            )
    text_column_name: str = field(
                                default="text",
                                metadata={
                                    "help": "The name of the dataset column containing the text data. Defaults to 'text'"
                                },
                            )
    max_duration_in_seconds: float = field(
                                default=20.0,
                                metadata={
                                    "help": (
                                        "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
                                        " 'max_duration_in_seconds`"
                                    )
                                },
                            )
    min_duration_in_seconds: float = field(
                                default=0.0,
                                metadata={
                                    "help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"
                                },
                            )
    preprocessing_only: bool = field(
                                default=False,
                                metadata={
                                    "help": (
                                        "Whether to only do data preprocessing and skip training. This is especially useful when data"
                                        " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                                        " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                                        " can consequently be loaded in distributed training"
                                    )
                                },
                            )
    language: str = field(
                                default=None,
                                metadata={
                                    "help": (
                                        "Language for multilingual fine-tuning. This argument should be set for multilingual fine-tuning "
                                        "only. For English speech recognition, it should be set to `None`."
                                    )
                                },
                            )

    data_dir: str = field(
                            default="/kaggle/input/bengaliai-speech",
                            metadata={
                                "help": (
                                    "Language for multilingual fine-tuning. This argument should be set for multilingual fine-tuning "
                                    "only. For English speech recognition, it should be set to `None`."
                                )
                            },
                        )
                        


logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((Config, Seq2SeqTrainingArguments))

    cfg, training_args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(cfg.model_name_or_path)

    config.update(
        {
            "forced_decoder_ids": cfg.forced_decoder_ids,
            "suppress_tokens": cfg.suppress_tokens,
        }
    )

    # SpecAugment for whisper models
    if getattr(config, "model_type", None) == "whisper":
        config.update({"apply_spec_augment": cfg.apply_spec_augment})

    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)

    raw_datasets = DatasetDict()

    data_dir = Path(cfg.data_dir)

    raw_ds = load_dataset("csv", data_files=str(data_dir / "train.csv"), split="train")

    def add_mp3_path(examples):
        return {
            "audio": [str(data_dir / f"train_mp3s/{id_}.mp3") for id_ in examples["id"]]
        }

    raw_ds = raw_ds.map(add_mp3_path, batched=True, num_proc=cfg.preprocessing_num_workers)
    raw_ds = raw_ds.train_test_split(
        test_size=0.2, seed=training_args.seed, shuffle=True
    )

    raw_datasets["train"] = raw_ds["train"]
    raw_datasets["validation"] = raw_ds["test"]

    if cfg.max_train_samples:
        raw_datasets["train"] = raw_datasets["train"].select(
            range(min(cfg.max_train_samples, len(raw_datasets["train"])))
        )

    if cfg.max_eval_samples:
        raw_datasets["validation"] = raw_datasets["validation"].select(
            range(min(cfg.max_eval_samples, len(raw_datasets["validation"])))
        )

    # cast to audio
    raw_datasets = raw_datasets.cast_column(
        cfg.audio_column_name,
        datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
    )

    if cfg.language is not None:
        # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
        tokenizer.set_prefix_tokens(language=cfg.language, task="transcribe")

    # Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = cfg.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = cfg.min_duration_in_seconds * feature_extractor.sampling_rate
    audio_column_name = cfg.audio_column_name
    num_workers = cfg.preprocessing_num_workers
    text_column_name = cfg.text_column_name
    model_input_name = feature_extractor.model_input_names[0]
    # if SpecAugment is used for whisper models, return attention_mask to guide the mask along time axis
    forward_attention_mask = (
        getattr(config, "model_type", None) == "whisper"
        and getattr(config, "apply_spec_augment", False)
        and getattr(config, "mask_time_prob", 0) > 0
    )

    def prepare_dataset(batch):
        # process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(
            sample["array"],
            sampling_rate=sample["sampling_rate"],
            return_attention_mask=forward_attention_mask,
        )
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]
        batch["input_length"] = len(sample["array"])
        if forward_attention_mask:
            batch["attention_mask"] = inputs.get("attention_mask")[0]

        # process targets
        input_str = batch[text_column_name]
        batch["labels"] = tokenizer(input_str).input_ids
        return batch

    with training_args.main_process_first(desc="dataset map pre-processing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            num_proc=cfg.preprocessing_num_workers,
            desc="preprocess train dataset",
        )

    # filter data that is shorter than min_input_length or longer than
    # max_input_length
    def is_audio_in_length_range(length):
        return length > min_input_length and length < max_input_length

    vectorized_datasets = vectorized_datasets.filter(
        is_audio_in_length_range,
        num_proc=num_workers,
        input_columns=["input_length"],
    )

    def save_chunks(ds, chunk_size, prefix):
        for i in range(0, len(ds), chunk_size):
            ii = min(i + chunk_size, len(ds))

            ds.select(range(i, ii)).to_parquet(f"{prefix}_{i}_to_{ii}.parquet")

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with `args.preprocessing_only` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step `args.preprocessing_only` can then be set to `False` to load the
    # cached dataset
    if cfg.preprocessing_only:
        cache = {k: v.cache_files for k, v in vectorized_datasets.items()}
        logger.info(f"Data preprocessing finished. Files cached at {cache}.")

        save_chunks(
            vectorized_datasets["train"], 1000, f"train_{training_args.output_dir}"
        )
        save_chunks(
            vectorized_datasets["validation"], 1000, f"eval_{training_args.output_dir}"
        )
        return


if __name__ == "__main__":
    main()

    