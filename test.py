import argparse
import torch
import logging
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Test the fine-tuned documentation optimizer model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./api-docs-model",
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument(
        "--original_code_file",
        type=str,
        help="Path to a file containing the original code",
    )
    parser.add_argument(
        "--updated_code_file",
        type=str,
        help="Path to a file containing the updated code",
    )
    parser.add_argument(
        "--original_doc_file",
        type=str,
        help="Path to a file containing the original documentation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum length of generated documentation",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Number of beams for beam search",
    )
    return parser.parse_args()

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def generate_improved_documentation(model, tokenizer, original_code, updated_code, original_doc, max_length=512, num_beams=5):
    # Format the input according to the training format
    instruction = f"""Improve the following Python documentation to align with the updated code:

Original code:
```python
{original_code}```
Updated code:
```python
{updated_code}```
Original documentation:
```python
{original_doc}```
Please provide an improved version of the documentation that reflects the code changes.
"""
    # Tokenize input
    inputs = tokenizer(instruction, return_tensors="pt", truncation=True, max_length=512).to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate improved documentation
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

    # Decode the generated output
    improved_doc = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return improved_doc


def main(): 
    args = parse_args()
    # Load model and tokenizer
    logger.info(f"Loading model and tokenizer from {args.model_path}")

    # Load the base model first
    config = PeftConfig.from_pretrained(args.model_path)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # Load the PEFT adapter
    model = PeftModel.from_pretrained(base_model, args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Move model to appropriate device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model.eval()

    # Read input files if provided
    if args.original_code_file and args.updated_code_file and args.original_doc_file:
        logger.info("Reading input files...")
        original_code = read_file(args.original_code_file)
        updated_code = read_file(args.updated_code_file)
        original_doc = read_file(args.original_doc_file)
    else:
        # Use example inputs if no files provided
        print("No input files provided. Using example inputs.")
        pass

    # Generate improved documentation
    logger.info("Generating improved documentation...")
    improved_doc = generate_improved_documentation(
        model,
        tokenizer,
        original_code,
        updated_code,
        original_doc,
        args.max_length,
        args.num_beams
    )

    # Print results
    logger.info("-" * 50)
    logger.info("ORIGINAL DOCUMENTATION:")
    logger.info(original_doc)
    logger.info("-" * 50)
    logger.info("IMPROVED DOCUMENTATION:")
    logger.info(improved_doc)
    logger.info("-" * 50)

if __name__ == "__main__":
    main()