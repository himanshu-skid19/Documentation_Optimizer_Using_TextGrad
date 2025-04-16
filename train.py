import os
import argparse
import logging
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from evaluate import load
import nltk
from nltk.tokenize import word_tokenize
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK data for evaluation metrics
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model for API documentation generation")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="api_docs_dataset",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="google/flan-t5-large",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./api-docs-model",
        help="Where to store the fine-tuned model",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=256,  # Reduce from 512 to 256
        help="Maximum length of the source text (instruction)",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=512,  # Reduce from 1024 to 512
        help="Maximum length of the target text (response)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Initial learning rate (after warmup period)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,  # Reduce from 4 to 1
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA attention dimension",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["4bit", "8bit", "none"],
        default="4bit",
        help="Quantization type",
    )
    return parser.parse_args()

def preprocess_function(examples, tokenizer, max_source_length, max_target_length):
    """Preprocess the data by tokenizing."""
    # Concatenate instruction and input
    inputs = examples["instruction"]
    targets = examples["response"]
    
    # Ensure inputs and targets are lists of strings
    if isinstance(inputs, str):
        inputs = [inputs]
    if isinstance(targets, str):
        targets = [targets]
    
    # Convert any non-string elements to strings
    inputs = [str(item) if item is not None else "" for item in inputs]
    targets = [str(item) if item is not None else "" for item in targets]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=max_source_length, 
        padding="max_length", 
        truncation=True
    )
    
    # Setup the tokenizer for targets
    labels = tokenizer(
        targets, 
        max_length=max_target_length, 
        padding="max_length", 
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    
    # Replace padding token id with -100 so they're ignored in the loss
    model_inputs["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in label] 
        for label in model_inputs["labels"]
    ]
    
    return model_inputs

global tokenizer

def compute_metrics(eval_preds):
    """Compute evaluation metrics."""
    rouge_score = load("rouge")
    bleu = load("bleu")
    
    preds, labels = eval_preds
    
    # Replace -100 with pad token id
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(word_tokenize(pred)) for pred in decoded_preds]
    decoded_labels = ["\n".join(word_tokenize(label)) for label in decoded_labels]
    
    # Compute ROUGE scores
    rouge_results = rouge_score.compute(
        predictions=decoded_preds, 
        references=decoded_labels, 
        use_stemmer=True
    )
    
    # Compute BLEU scores
    tokenized_preds = [word_tokenize(pred) for pred in decoded_preds]
    tokenized_labels = [[word_tokenize(label)] for label in decoded_labels]
    bleu_results = bleu.compute(predictions=tokenized_preds, references=tokenized_labels)
    
    # Extract the scores we want to track
    result = {
        'rouge1': rouge_results['rouge1'],
        'rouge2': rouge_results['rouge2'],
        'rougeL': rouge_results['rougeL'],
        'bleu': bleu_results['bleu'],
    }
    
    return result

def main():
    args = parse_args()
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Add padding token if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup quantization config
    if args.quantization == "4bit":
        logger.info("Using 4-bit quantization")
        compute_dtype = torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.quantization == "8bit":
        logger.info("Using 8-bit quantization")
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        logger.info("Not using quantization")
        quant_config = None
    
    # Load model with quantization config
    logger.info(f"Loading model: {args.model_name_or_path}")
    if args.quantization in ["4bit", "8bit"]:
        # For quantized models, don't use any device mapping or dispatch
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=quant_config,
            device_map=None,  
            torch_dtype=torch.float16,
            # Add these params to prevent device movement
            _fast_init=False,
            low_cpu_mem_usage=True
        )
    else:
        # For non-quantized models
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto"
        )
    
    # Prepare model for k-bit training
    if args.quantization in ["4bit", "8bit"]:
        logger.info("Preparing model for quantized training")
        model = prepare_model_for_kbit_training(model)
        
        # Enable gradient checkpointing here
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # Important when using gradient checkpointing
    
    # Setup LoRA
    logger.info("Setting up LoRA for efficient fine-tuning")
    
    # Define target modules for T5 models
    target_modules = ["q", "v"]  # For T5 models
    
    # Define LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)

    # Add this line to mark trainable parameters
    for param in model.parameters():
        if param.requires_grad:
            # Add a small update to activate gradients properly
            param.data = param.data.to(torch.float32)
            
    model.print_trainable_parameters()
    # Preprocess datasets
    logger.info("Preprocessing datasets")
        
    tokenized_train = dataset["train"].map(
        lambda examples: preprocess_function(
            examples, tokenizer, args.max_source_length, args.max_target_length
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    tokenized_val = dataset["validation"].map(
        lambda examples: preprocess_function(
            examples, tokenizer, args.max_source_length, args.max_target_length
        ),
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )
    
    # Define training arguments - reduced batch size and gradient accumulation steps
    # to fit within memory constraints
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,  # Increase from 4 to 8
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        gradient_checkpointing=False,  # Try disabling this first
        fp16=True,  # Use mixed precision
        ddp_find_unused_parameters=False,  # Add this parameter
    )
    
    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the model and tokenizer
    logger.info(f"Saving model and tokenizer to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()