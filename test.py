import os
import argparse
import logging
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from evaluate import load
import torch
import nltk
from train import preprocess_function, preprocess_codocbench, load_codocbench_dataset  # Reuse preprocessing functions
from peft import PeftModel, PeftConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure nltk tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned model on CoDocBench test set")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the fine-tuned model and tokenizer"
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="processed_dataset/test.jsonl",
        help="Path to the test dataset file"
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=512,
        help="Maximum input length"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=512,
        help="Maximum target length"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--codocbench_train",
        type=str,
        default="processed_dataset/train.jsonl",
        help="Path to the CoDocBench training dataset file",
    )
    parser.add_argument(
        "--codocbench_test",
        type=str,
        default="processed_dataset/test.jsonl",
        help="Path to the CoDocBench test dataset file",
    )
    parser.add_argument(
        "--codocbench_val",
        type=str,
        default="processed_dataset/val.jsonl",
        help="Path to the CoDocBench validation dataset file",
    )
    return parser.parse_args()

def compute_metrics(eval_preds, tokenizer):
    """Compute evaluation metrics."""
    rouge_score = load("rouge")
    bleu = load("bleu")

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    pad_id = tokenizer.pad_token_id or 0
    max_id = tokenizer.vocab_size - 1

    preds = np.clip(preds, 0, max_id).astype(np.int32)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, pad_id)
    labels = np.clip(labels, 0, max_id).astype(np.int32)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge_results = rouge_score.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    bleu_results = bleu.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        'rouge1': rouge_results['rouge1'],
        'rouge2': rouge_results['rouge2'],
        'rougeL': rouge_results['rougeL'],
        'bleu': bleu_results['bleu'],
    }

def main():
    args = parse_args()

    logger.info(f"Loading model from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # Load LoRA adapter + base model
    config = PeftConfig.from_pretrained(args.model_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, args.model_dir)

    logger.info(f"Loading test dataset from {args.test_path}")
    dataset = load_codocbench_dataset(args.codocbench_train, args.codocbench_val, args.codocbench_test)
    processed_test = dataset['test']
    logger.info("Tokenizing test data")
    tokenized_test = processed_test.map(
        lambda examples: preprocess_function(
            examples, tokenizer, args.max_source_length, args.max_target_length
        ),
        batched=True,
        remove_columns=processed_test.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="./test_output",
        per_device_eval_batch_size=args.batch_size,
        do_train=False,
        do_eval=True,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        logging_dir="./logs",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        eval_dataset=tokenized_test,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )

    logger.info("Evaluating model...")
    metrics = trainer.evaluate()
    logger.info("Test Evaluation Metrics:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()
