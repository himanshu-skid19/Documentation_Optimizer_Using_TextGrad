import os
import argparse
import logging
import numpy as np
from datasets import load_from_disk, load_dataset
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
import wandb  # Add wandb import


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

# At the module level (outside of any function)
tokenizer = None

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model for API documentation generation")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="codocbench/dataset/",
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
        default=512,  # Reduce from 512 to 256
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
        default=1e-3,
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
        default=32,
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
        default=0.1,
        help="LoRA dropout probability",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["4bit", "8bit", "none"],
        default="4bit",
        help="Quantization type",
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
    # Add WandB related arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="doc-optimizer",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="WandB entity (username or team name)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="WandB run name",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to log to Weights & Biases",
    )
    return parser.parse_args()

def load_codocbench_dataset(train_path, val_path, test_path):
    """Load the CoDocBench dataset from JSONL files."""
    
    # Load the datasets
    train_dataset = load_dataset('json', data_files=train_path, split='train')
    val_dataset = load_dataset('json', data_files=train_path, split='train')
    test_dataset = load_dataset('json', data_files=test_path, split='train')
    
    # Combine into a single dataset dict
    dataset = {
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    }
    
    return dataset

def preprocess_codocbench(examples):
    """Transform CoDocBench data into instruction/response pairs."""
    instructions = []
    responses = []
    
    for i in range(len(examples['file'])):
        # Get the two versions
        version_data = examples['version_data'][i]
        if len(version_data) < 2:
            continue
            
        v1 = version_data[0]
        v2 = version_data[1]
        
        # Create instruction based on code changes
        instruction = f"""Improve the following Python documentation to align with the updated code:

    Original code:
    ```python
    {v1['code']}```
    Updated code:
    ```python
    {v2['code']}
    ```
    Original documentation:
    ```python
    {v1['docstring']}
    ```
    Please provide an improved version of the documentation that reflects the code changes."""
            # Use the updated docstring as the response
    response = v2['docstring']
    
    instructions.append(instruction)
    responses.append(response)

    return {
        "instruction": instructions,
        "response": responses
    }


    



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

def compute_metrics(eval_preds):
    """Compute evaluation metrics."""
    global tokenizer  # Reference the global tokenizer
    rouge_score = load("rouge")
    bleu = load("bleu")
    
    preds, labels = eval_preds
    
    # Process predictions and labels (keep this part as is)
    if isinstance(preds, tuple):
        preds = preds[0]
    
    max_id = tokenizer.vocab_size - 1
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds)
        
    preds = np.clip(preds, 0, max_id)
    preds = preds.astype(np.int32)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    labels = np.where(labels != -100, labels, pad_id)
    labels = np.clip(labels, 0, max_id).astype(np.int32)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE calculation (keep this part as is)
    rouge_results = rouge_score.compute(
        predictions=decoded_preds, 
        references=decoded_labels, 
        use_stemmer=True
    )
    
    # FIX: BLEU calculation - use strings directly, not tokenized words
    bleu_results = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    
    # Extract the scores we want to track
    result = {
        'rouge1': rouge_results['rouge1'],
        'rouge2': rouge_results['rouge2'],
        'rougeL': rouge_results['rougeL'],
        'bleu': bleu_results['bleu'],
    }
    
    return result


class DebugTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        if labels is not None:
            print("Batch Labels (non -100):", (labels != -100).sum().item())
        outputs = model(**inputs)
        loss = outputs.loss
        print("Computed loss:", loss.item())
        return (loss, outputs) if return_outputs else loss

def main():
    args = parse_args()
    
    # Initialize WandB
    if args.use_wandb:
        logger.info(f"Initializing Weights & Biases with project: {args.wandb_project}")
        wandb_run_name = args.wandb_run_name or f"{args.model_name_or_path.split('/')[-1]}-{args.quantization}"
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            config={
                "model": args.model_name_or_path,
                "max_source_length": args.max_source_length,
                "max_target_length": args.max_target_length,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "epochs": args.num_epochs,
                "quantization": args.quantization,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
            }
        )
    
    logger.info(f"Loading CoDocBench dataset from {args.codocbench_train} and {args.codocbench_test}")
    dataset = load_codocbench_dataset(args.codocbench_train, args.codocbench_val, args.codocbench_test)

    # processed_train = dataset['train'].map( preprocess_codocbench, batched=True, remove_columns=dataset['train'].column_names, )
    # processed_val = dataset['validation'].map( preprocess_codocbench, batched=True, remove_columns=dataset['validation'].column_names, )
    processed_train = dataset['train']
    processed_val = dataset['validation']
    global tokenizer
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

    # # Add this line to mark trainable parameters
    # for param in model.parameters():
    #     if param.requires_grad:
    #         # Add a small update to activate gradients properly
    #         param.data = param.data.to(torch.float32)
            
    model.print_trainable_parameters()
    # Preprocess datasets
    logger.info("Preprocessing datasets")
        
    tokenized_train = processed_train.map(
        lambda examples: preprocess_function(
            examples, tokenizer, args.max_source_length, args.max_target_length
        ),
        batched=True,
        remove_columns=processed_train.column_names,
    )
    
    tokenized_val = processed_val.map(
        lambda examples: preprocess_function(
            examples, tokenizer, args.max_source_length, args.max_target_length
        ),
        batched=True,
        remove_columns=processed_val.column_names,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    # num_tokens = sum([sum(1 for t in ex['labels'] if t != -100) for ex in tokenized_train])
    # print(f"Total non-ignored tokens in training set: {num_tokens}")

        
    # Define training arguments - reduced batch size and gradient accumulation steps
    # to fit within memory constraints
    report_to = ["wandb", "tensorboard"] if args.use_wandb else ["tensorboard"]
    
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
        report_to=report_to,  # Updated to include wandb when enabled
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        gradient_checkpointing=False,  # Try disabling this first
        fp16=False,  # Use mixed precision
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

    # trainer = DebugTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_train,
    #     eval_dataset=tokenized_val,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics,
    # )
    
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.shape} | mean={param.data.mean():.4f}")

    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the model and tokenizer
    logger.info(f"Saving model and tokenizer to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Finish WandB run
    if args.use_wandb:
        wandb.finish()
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()