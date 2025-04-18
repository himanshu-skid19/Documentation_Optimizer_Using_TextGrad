import pytest
import os
import torch
import tempfile
from unittest.mock import patch, MagicMock
import sys
from Documentation_Optimizer_Using_TextGrad.test import parse_args, read_file, generate_improved_documentation
from Documentation_Optimizer_Using_TextGrad.test import main

# Import functions from the module using absolute import

def test_read_file():
    """Test that read_file properly reads file content."""
    # Create a temporary file with known content
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        temp_file.write("test content")
        temp_file_path = temp_file.name
    
    try:
        # Test reading from the file
        result = read_file(temp_file_path)
        assert result == "test content"
    finally:
        # Clean up
        os.unlink(temp_file_path)

@patch('sys.argv', ['test.py', 
                   '--model_path=./test-model',
                   '--original_code_file=orig_code.py',
                   '--updated_code_file=new_code.py',
                   '--original_doc_file=orig_doc.txt',
                   '--max_length=256',
                   '--num_beams=3'])
def test_parse_args():
    """Test that argument parsing works properly."""
    args = parse_args()
    assert args.model_path == "./test-model"
    assert args.original_code_file == "orig_code.py"
    assert args.updated_code_file == "new_code.py"
    assert args.original_doc_file == "orig_doc.txt"
    assert args.max_length == 256
    assert args.num_beams == 3

def test_generate_improved_documentation():
    """Test documentation generation with mocked model and tokenizer."""
    # Create mock objects
    model = MagicMock()
    tokenizer = MagicMock()
    
    # Configure tokenizer mock
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }
    tokenizer.decode.return_value = "Improved documentation"
    
    # Configure model mock
    model.generate.return_value = torch.tensor([[4, 5, 6]])
    
    # Test input data
    original_code = "def old_function():\n    return None"
    updated_code = "def old_function(param):\n    return param"
    original_doc = "This function returns None."
    
    # Patch torch.cuda.is_available
    with patch('torch.cuda.is_available', return_value=False):
        # Call the function
        result = generate_improved_documentation(
            model,
            tokenizer,
            original_code,
            updated_code,
            original_doc,
            max_length=100,
            num_beams=2
        )
    
    # Verify the result
    assert result == "Improved documentation"
    
    # Verify the tokenizer was called with the expected instruction format
    instruction_arg = tokenizer.call_args[0][0]
    assert "Original code:" in instruction_arg
    assert "Updated code:" in instruction_arg
    assert "Original documentation:" in instruction_arg
    
    # Verify model.generate was called
    model.generate.assert_called_once()
    
    # Verify tokenizer.decode was called with the model's output
    tokenizer.decode.assert_called_once_with(torch.tensor([4, 5, 6]), skip_special_tokens=True)

@patch('Documentation_Optimizer_Using_TextGrad.test.PeftConfig')
@patch('Documentation_Optimizer_Using_TextGrad.test.AutoModelForSeq2SeqLM')
@patch('Documentation_Optimizer_Using_TextGrad.test.PeftModel')
@patch('Documentation_Optimizer_Using_TextGrad.test.AutoTokenizer')
@patch('Documentation_Optimizer_Using_TextGrad.test.generate_improved_documentation')
@patch('Documentation_Optimizer_Using_TextGrad.test.read_file')
@patch('Documentation_Optimizer_Using_TextGrad.test.parse_args')
def test_main(mock_parse_args, mock_read_file, mock_gen_doc, mock_tokenizer, 
              mock_peft, mock_auto_model, mock_peft_config):
    """Test the main function with mocked components."""
    
    # Configure mocks
    mock_parse_args.return_value = MagicMock(
        model_path="./model",
        original_code_file="old.py",
        updated_code_file="new.py",
        original_doc_file="doc.txt",
        max_length=256,
        num_beams=4
    )
    
    mock_read_file.side_effect = ["original code", "updated code", "original doc"]
    mock_gen_doc.return_value = "improved documentation"
    
    # Run the main function
    with patch('torch.cuda.is_available', return_value=False):
        main()
    
    # Verify interactions
    mock_read_file.assert_any_call("old.py")
    mock_read_file.assert_any_call("new.py")
    mock_read_file.assert_any_call("doc.txt")
    
    mock_gen_doc.assert_called_once()