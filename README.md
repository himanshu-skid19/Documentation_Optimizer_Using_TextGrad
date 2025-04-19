# Documentation_Optimizer_Using_TextGrad


This is done as a course project for the course DA312: Advanced Machine Learning Laboratory.

## Overview
This project implements a documentation optimization system that iteratively enhances API documentation by evaluating it across multiple quality metrics and suggesting improvements. It leverages TextGrad, a framework that enables gradient-based optimization of text using LLMs.

## Features
- Multi-metric Optimization: Evaluates documentation based on completeness, accuracy, and usability
- Multiple Model Support:
OpenAI models (GPT-4o-mini)
Local models via LMStudio
LlamaCPP models
- Interactive Web UI: Streamlit-based interface for easy documentation optimization
- Optimization History: Track changes throughout the optimization process
- Customizable Parameters: Configure iterations, temperature, and model settings

## How It Works
1. The system takes API documentation as input
2. TextGrad creates a variable representing the documentation
3. An LLM evaluates the documentation quality across metrics
4. TextGrad computes gradients based on the evaluation
5. The documentation is updated using gradient descent
6. The process repeats for the specified number of iterations

## Installation

```python
# Clone the repository
git clone https://github.com/himanshu-skid19/Documentation_Optimizer_Using_TextGrad.git
cd Documentation_Optimizer_Using_TextGrad

# Create a virtual environment
conda create --name <env_name> python=3.10

# Activate the virtual env
conda activate <env_name>

# Install dependencies
pip install -r requirements.txt
```

## Usage
#### Command Line Interface
```bash
python inference.py --doc "path/to/documentation.md" --model "openai" --iterations 3
```

#### Web Interface
```bash
python -m streamlit run demo.py
```
The web interface allows you to:

- Select model type (OpenAI or Local via LMStudio)
- Enter or upload API documentation
- Configure optimization parameters
- View step-by-step optimization progress
- Track optimization history

### Configuration Options
```python
APIDocOptimizer(
    model_name="llama",           # Model type: "llama", "openai", etc.
    model_path="/path/to/model",  # Path to local model weights
    openai_api_key=None,          # OpenAI API key (or set via environment)
    use_local_model=False,        # Whether to use a local model via LMStudio
    iterations=3,                 # Number of optimization iterations
    temperature=0.1,              # Temperature for LLM generation
    verbose=True                  # Whether to print detailed output
)
```
Example
```python 
from inference import APIDocOptimizer

# Initialize optimizer
optimizer = APIDocOptimizer(
    model_name="openai",
    openai_api_key="your-api-key-here"
)

# Sample documentation
documentation = """
# getUserProfile(id)
Gets user profile data.
"""

# Optimize documentation
improved_doc = optimizer.optimize(documentation)
print(improved_doc)
```

Credits
Built with TextGrad framework.