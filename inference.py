import os
import argparse
import json
from typing import Dict, List, Any, Optional, Tuple

# TextGrad imports
import textgrad as tg
from textgrad.engine_experimental.openai import OpenAIEngine

from dotenv import load_dotenv
from textgrad.engine import get_engine
load_dotenv()

class APIDocOptimizer:
    """API Documentation Optimizer using TextGrad framework.
    
    This class implements an optimizer that takes API documentation as input
    and iteratively improves it using the TextGrad framework with LLMs.
    """
    
    def __init__(
        self, 
        model_name: str = "llama",
        model_path: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        use_local_model: bool = False,
        iterations: int = 3,
        temperature: float = 0.1,
        verbose: bool = True
    ):
        """Initialize the API Documentation Optimizer.
        
        Args:
            model_name: The name of the model to use ('llama' or 'openai')
            model_path: Path to local model (for llama model)
            openai_api_key: OpenAI API key (for OpenAI models)
            use_local_model: Whether to use a local model with LMStudio
            iterations: Number of optimization iterations
            temperature: Temperature parameter for text generation
            verbose: Whether to print verbose output
        """
        self.model_name = model_name
        self.model_path = model_path
        self.openai_api_key = openai_api_key
        self.use_local_model = use_local_model
        self.iterations = iterations
        self.temperature = temperature
        self.verbose = verbose
        
        # Initialize the TextGrad engine
        self._initialize_engine()
        
        # Define optimization metrics
        self.metrics = [
            "completeness",  # Coverage of essential elements
            "accuracy",      # Technical accuracy of information
            "usability"      # Clarity and practicality of examples
        ]
    
    def _initialize_engine(self):
        """Initialize the TextGrad engine with the appropriate LLM."""
        if self.use_local_model:
            # Use local model with LMStudio as in the notebook example
            try:
                from openai import OpenAI
                from textgrad.engine.local_model_openai_api import ChatExternalClient
                
                # Use default LMStudio address
                client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
                
                # Specify the model string appropriately
                model_string = self.model_path or "local-model"
                self.engine = ChatExternalClient(client=client, model_string=model_string)
                
            except ImportError:
                print("Error: Required packages for local model not found.")
                print("Install with: pip install openai textgrad")
                raise
                
        elif self.model_name.lower() == "llama":
            # Use LlamaCPP from LlamaIndex
            try:
                if not self.model_path:
                    raise ValueError("model_path is required for llama model")
                
                # Initialize LlamaCPP model
                llm = LlamaCPP(
                    model_path=self.model_path,
                    temperature=self.temperature,
                    max_new_tokens=2048,
                    context_window=4096,
                    generate_kwargs={"temperature": self.temperature},
                    model_kwargs={"n_gpu_layers": -1}
                )
                
                # Create a TextGrad engine from the LLM
                self.engine = tg.CustomModelClient(
                    model=llm,
                    temperature=self.temperature
                )
                
            except ImportError:
                print("Error: Required packages for Llama model not found.")
                print("Install with: pip install llama-index llama-cpp-python textgrad")
                raise
                
        elif self.model_name.lower() == "openai":
            # Use OpenAI models
            if not self.openai_api_key:
                self.openai_api_key = os.environ.get("OPENAI_API_KEY")
                if not self.openai_api_key:
                    raise ValueError("OpenAI API key is required for OpenAI models")
            
            # Set the API key
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
            
            # Initialize the OpenAI engine
            self.engine = get_engine("experimental:gpt-4o-mini", cache=False)
        
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
        # Set the backward engine for TextGrad
        tg.set_backward_engine(self.engine, override=True)
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the documentation evaluation."""
        return """You are an expert API documentation reviewer. 
        Evaluate the given API documentation for the following criteria:
        1. Completeness: Does it cover all essential elements (description, parameters, return values, examples)?
        2. Technical Accuracy: Is the information correct and precise?
        3. Usability: Is the documentation clear, practical, and helpful for developers?
        
        Identify specific problems and suggest improvements. Be detailed and constructive in your feedback.
        Focus on making the documentation more helpful for developers using the API.
        """
    
    def optimize(self, documentation: str) -> str:
        """Optimize the given API documentation using TextGrad.
        
        Args:
            documentation: The API documentation to optimize
            
        Returns:
            The optimized API documentation
        """
        if self.verbose:
            print("Starting optimization process...")
            print(f"Original documentation:\n{documentation}\n")
        
        # Create the TextGrad variable for the documentation
        doc_var = tg.Variable(
            documentation,
            requires_grad=True,
            role_description="API documentation"
        )
        
        # Create the system prompt for evaluation
        system_prompt = tg.Variable(
            self._create_system_prompt(),
            requires_grad=False,
            role_description="system prompt"
        )
        
        # Create the loss function
        loss_fn = tg.TextLoss(system_prompt)
        
        # Create the optimizer
        optimizer = tg.TGD([doc_var])
        
        # Optimization loop
        for i in range(self.iterations):
            if self.verbose:
                print(f"\nIteration {i+1}/{self.iterations}")
            
            # Compute the loss (evaluation)
            loss = loss_fn(doc_var)
            
            if self.verbose:
                print(f"Evaluation feedback:\n{loss.value}")
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Update the documentation
            optimizer.step()
            
            if self.verbose:
                print(f"Updated documentation:\n{doc_var.value}")
        
        # Return the optimized documentation
        return doc_var.value

    def evaluate_documentation(self, documentation: str) -> Dict[str, float]:
        """Evaluate the documentation and return scores for each metric.
        
        Args:
            documentation: The API documentation to evaluate
            
        Returns:
            Dictionary with scores for each metric (0-10 scale)
        """
        # Create an evaluation prompt
        eval_prompt = f"""
        Please evaluate the following API documentation on a scale of 0-10 for each of these criteria:
        - Completeness: Coverage of essential elements (description, parameters, returns, examples)
        - Technical Accuracy: Correctness of type information and parameter descriptions
        - Usability: Clarity and practicality of examples and explanations
        
        Documentation to evaluate:
        ```
        {documentation}
        ```
        
        Return your evaluation as a JSON object with the format:
        {{
            "completeness": score,
            "accuracy": score,
            "usability": score,
            "overall": average_score,
            "feedback": "detailed feedback here"
        }}
        """
        
        # Create a TextGrad variable for the documentation to evaluate
        eval_var = tg.Variable(
            eval_prompt,
            requires_grad=False,
            role_description="evaluation request"
        )
        
        # Get the evaluation response
        response = self.engine.generate(content="Please evaluate the documentation according to the criteria provided.",system_prompt=eval_prompt)
        
        # Extract JSON from the response
        try:
            # Find JSON-like content in the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                scores = json.loads(json_str)
            else:
                # Fallback if JSON parsing fails
                scores = {
                    "completeness": 0,
                    "accuracy": 0,
                    "usability": 0,
                    "overall": 0,
                    "feedback": "Failed to parse evaluation"
                }
        except json.JSONDecodeError:
            scores = {
                "completeness": 0,
                "accuracy": 0,
                "usability": 0, 
                "overall": 0,
                "feedback": "Failed to parse evaluation response"
            }
        
        return scores


def main():
    """Main function to demonstrate the API Documentation Optimizer."""
    parser = argparse.ArgumentParser(description="API Documentation Optimizer")
    parser.add_argument("--input", type=str, help="Input documentation file or string")
    parser.add_argument("--model", type=str, default="openai", help="Model to use (llama or openai)")
    parser.add_argument("--model-path", type=str, help="Path to local model (for llama)")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (for OpenAI models)")
    parser.add_argument("--local", action="store_true", help="Use local model with LMStudio")
    parser.add_argument("--iterations", type=int, default=3, help="Number of optimization iterations")
    parser.add_argument("--output", type=str, help="Output file for optimized documentation")
    args = parser.parse_args()
    
    # Example documentation if none provided
    example_doc = """
    Description
    Create a new user.

    Parameters
    name: User's name
    age: User's age
    email: User's email

    Returns
    The created user data

    Examples
    user = create_user(name="John", age=25)
    """
    
    # Get the input documentation
    if args.input:
        if os.path.isfile(args.input):
            with open(args.input, 'r') as f:
                documentation = f.read()
        else:
            documentation = args.input
    else:
        documentation = example_doc
    
    # Create the optimizer
    optimizer = APIDocOptimizer(
        model_name=args.model,
        model_path=args.model_path,
        openai_api_key=args.api_key,
        use_local_model=args.local,
        iterations=args.iterations
    )
    
    # Evaluate initial documentation
    print("Evaluating initial documentation...")
    initial_scores = optimizer.evaluate_documentation(documentation)
    print(f"Initial scores: {initial_scores}")
    
    # Optimize the documentation
    print("\nOptimizing documentation...")
    optimized_doc = optimizer.optimize(documentation)
    
    # Evaluate optimized documentation
    print("\nEvaluating optimized documentation...")
    final_scores = optimizer.evaluate_documentation(optimized_doc)
    print(f"Final scores: {final_scores}")
    
    # Print the optimized documentation
    print("\nOptimized Documentation:")
    print(optimized_doc)
    
    # Save to output file if specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write(optimized_doc)
        print(f"\nOptimized documentation saved to {args.output}")


if __name__ == "__main__":
    main()