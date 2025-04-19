import os
import argparse
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import diskcache as dc

# TextGrad imports
import textgrad as tg
from textgrad import Variable
from textgrad.optimizer import TextualGradientDescent as TGD
from textgrad.loss import TextLoss
from textgrad.engine import get_engine
from textgrad.engine_experimental.openai import OpenAIEngine

# PEFT and Transformer imports
try:
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# OpenAI imports
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# LMStudio imports
try:
    from textgrad.engine.local_model_openai_api import ChatExternalClient
    LMSTUDIO_AVAILABLE = True
except ImportError:
    LMSTUDIO_AVAILABLE = False

# Llama imports
try:
    from llama_index.llms import LlamaCPP
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

# ========================== MODEL ADAPTERS ==========================

class PeftModelAdapter:
    """Adapter to make PEFT models compatible with TextGrad's interface."""
    
    def __init__(self, model_dir, temperature=0.1):
        """Initialize the PEFT model adapter."""
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT and Transformers libraries are required. Install with: pip install peft transformers torch")
            
        # Load PEFT model using the correct sequence
        self.config = PeftConfig.from_pretrained(model_dir)
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(self.config.base_model_name_or_path)
        self.model = PeftModel.from_pretrained(self.base_model, model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name_or_path)
        self.temperature = temperature
        
    def generate(self, content, system_prompt=None):
        """Generate text based on input content following TextGrad's interface."""
        # Combine system prompt and content if provided
        if system_prompt:
            input_text = f"{system_prompt}\n\n{content}"
        else:
            input_text = content
        
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate response
        with torch.no_grad():
            # Create a dictionary of generation kwargs and pass it using **kwargs syntax
            generation_kwargs = {
                "input_ids": inputs["input_ids"],
                "max_new_tokens": 512,
                "do_sample": self.temperature > 0,
                "temperature": self.temperature if self.temperature > 0 else None
            }
            
            # This ensures all args are passed as keyword arguments
            outputs = self.model.generate(**generation_kwargs)
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

# ========================== ENGINE IMPLEMENTATIONS ==========================

class PeftEngine:
    """Custom TextGrad engine implementation for PEFT models."""
    
    def __init__(self, model_dir, temperature=0.1, use_cache=True, cache_dir=".textgrad_cache"):
        """Initialize the PEFT engine.
        
        Args:
            model_dir: Directory containing the PEFT model
            temperature: Temperature for generation
            use_cache: Whether to cache responses
            cache_dir: Directory to store cached responses
        """
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT and Transformers libraries are required. Install with: pip install peft transformers torch")
            
        # Create the PEFT model adapter
        self.model = PeftModelAdapter(model_dir=model_dir, temperature=temperature)
        self.temperature = temperature
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.name = f"PeftEngine({os.path.basename(model_dir)})"
        
        # Initialize cache if needed
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache = dc.Cache(self.cache_dir)
            
    def _process_textgrad_variable(self, var):
        """Helper method to extract value from TextGrad variables."""
        if hasattr(var, 'value'):
            return var.value
        return var
    
    def _hash_prompt(self, content: str, system_prompt: Optional[str] = None):
        """Create a hash for the prompt to use as cache key."""
        combined = f"{system_prompt if system_prompt else ''}|||{content}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def generate(self, content: str, system_prompt: Optional[str] = None, **kwargs):
        """Generate text using the PEFT model."""
        content_text = self._process_textgrad_variable(content)
        system_prompt_text = self._process_textgrad_variable(system_prompt) if system_prompt else None
    
        # Check cache first if enabled
        if self.use_cache:
            cache_key = self._hash_prompt(content_text, system_prompt_text)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Generate response using the model
        response = self.model.generate(content=content_text, system_prompt=system_prompt_text)
        
        # Cache the result if enabled
        if self.use_cache:
            self.cache[cache_key] = response
        
        return response
    
    def __call__(self, content: str, system_prompt: Optional[str] = None, **kwargs):
        """Make the engine callable, same as generate."""
        return self.generate(content=content, system_prompt=system_prompt, **kwargs)
    
    def __str__(self):
        return self.name


class PeftBackwardEngine(PeftEngine):
    """Backward engine implementation for TextGrad's gradient computation."""
    
    def __init__(self, model_dir, temperature=0.1, use_cache=True, cache_dir=".textgrad_cache"):
        """Initialize the PEFT backward engine."""
        super().__init__(model_dir, temperature, use_cache, cache_dir)
        self.name = f"PeftBackwardEngine({os.path.basename(model_dir)})"

    def backward(self, output, content=None, system_prompt=None, **kwargs):
        """Generate gradients for TextGrad optimization."""
        # Process the inputs
        if isinstance(output, str):
            output_text = output
        else:
            output_text = output.value if hasattr(output, 'value') else str(output)
        
        if content is None:
            return {"gradient": "This is a new improved documentation."}
        
        content_text = content.value if hasattr(content, 'value') else str(content)
        
        # Create a much more specific prompt that will generate usable gradient text
        gradient_prompt = f"""
        You are a documentation improvement expert. Your task is to REWRITE the API documentation 
        to directly address the issues mentioned in the evaluation.
        
        Current documentation:
        ```
        {content_text}
        ```
        
        Evaluation of current documentation:
        ```
        {output_text}
        ```
        
        IMPORTANT: Don't explain what changes to make. Instead, provide the COMPLETE REVISED DOCUMENTATION
        with all improvements already implemented. The output should be ONLY the updated documentation text
        that can directly replace the original.
        
        Generate the improved documentation now:
        """
        
        # Generate the improved documentation directly
        improved_doc = self.model.generate(content=gradient_prompt, system_prompt=None)
        
        # TextGrad expects a dict with a "gradient" key
        # The value should be text that can directly improve the original content
        return {"gradient": improved_doc}


class LMStudioEngine:
    """TextGrad engine implementation for LMStudio models."""
    
    def __init__(self, base_url="http://localhost:1234/v1", api_key="lm-studio", model_string=None, temperature=0.1, use_cache=True, cache_dir=".textgrad_cache"):
        """Initialize the LMStudio engine.
        
        Args:
            base_url: URL of the LMStudio API
            api_key: API key for LMStudio
            model_string: Name of the model to use
            temperature: Temperature for generation
            use_cache: Whether to cache responses
            cache_dir: Directory to store cached responses
        """
        if not LMSTUDIO_AVAILABLE:
            raise ImportError("OpenAI and textgrad's local_model_openai_api module are required. Install with: pip install openai textgrad")
            
        # Initialize LMStudio client
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_string = model_string or "default-model"
        self.temperature = temperature
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.name = f"LMStudioEngine({self.model_string})"
        
        # Create the ChatExternalClient
        self.engine = ChatExternalClient(client=self.client, model_string=self.model_string)
        
        # Initialize cache if needed
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache = dc.Cache(self.cache_dir)
    
    def _process_textgrad_variable(self, var):
        """Helper method to extract value from TextGrad variables."""
        if hasattr(var, 'value'):
            return var.value
        return var
    
    def _hash_prompt(self, content: str, system_prompt: Optional[str] = None):
        """Create a hash for the prompt to use as cache key."""
        combined = f"{system_prompt if system_prompt else ''}|||{content}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def generate(self, content: str, system_prompt: Optional[str] = None, **kwargs):
        """Generate text using the LMStudio model."""
        content_text = self._process_textgrad_variable(content)
        system_prompt_text = self._process_textgrad_variable(system_prompt) if system_prompt else None
    
        # Check cache first if enabled
        if self.use_cache:
            cache_key = self._hash_prompt(content_text, system_prompt_text)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Generate response using the engine
        response = self.engine.generate(content=content_text, system_prompt=system_prompt_text)
        
        # Cache the result if enabled
        if self.use_cache:
            self.cache[cache_key] = response
        
        return response
    
    def __call__(self, content: str, system_prompt: Optional[str] = None, **kwargs):
        """Make the engine callable, same as generate."""
        return self.generate(content=content, system_prompt=system_prompt, **kwargs)
    
    def __str__(self):
        return self.name
    
    def backward(self, output, content=None, system_prompt=None, **kwargs):
        """Generate gradients for TextGrad optimization."""
        # Process the inputs
        if isinstance(output, str):
            output_text = output
        else:
            output_text = output.value if hasattr(output, 'value') else str(output)
        
        if content is None:
            return {"gradient": "This is a new improved documentation."}
        
        content_text = content.value if hasattr(content, 'value') else str(content)
        
        # Create a specific prompt for gradient generation
        gradient_prompt = f"""
        You are a documentation improvement expert. Your task is to REWRITE the API documentation 
        to directly address the issues mentioned in the evaluation.
        
        Current documentation:
        ```
        {content_text}
        ```
        
        Evaluation of current documentation:
        ```
        {output_text}
        ```
        
        IMPORTANT: Don't explain what changes to make. Instead, provide the COMPLETE REVISED DOCUMENTATION
        with all improvements already implemented. The output should be ONLY the updated documentation text
        that can directly replace the original.
        
        Generate the improved documentation now:
        """
        
        # Generate the improved documentation directly
        improved_doc = self.generate(content=gradient_prompt, system_prompt=None)
        
        # TextGrad expects a dict with a "gradient" key
        return {"gradient": improved_doc}


class LlamaCPPEngine:
    """TextGrad engine implementation for LlamaCPP models."""
    
    def __init__(self, model_path, temperature=0.1, use_cache=True, cache_dir=".textgrad_cache"):
        """Initialize the LlamaCPP engine.
        
        Args:
            model_path: Path to the LlamaCPP model
            temperature: Temperature for generation
            use_cache: Whether to cache responses
            cache_dir: Directory to store cached responses
        """
        if not LLAMACPP_AVAILABLE:
            raise ImportError("llama-index and llama-cpp-python are required. Install with: pip install llama-index llama-cpp-python")
            
        # Initialize LlamaCPP
        self.llm = LlamaCPP(
            model_path=model_path,
            temperature=temperature,
            max_new_tokens=2048,
            context_window=4096,
            generate_kwargs={"temperature": temperature},
            model_kwargs={"n_gpu_layers": -1}
        )
        
        self.temperature = temperature
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.name = f"LlamaCPPEngine({os.path.basename(model_path)})"
        
        # Initialize cache if needed
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache = dc.Cache(self.cache_dir)
    
    def _process_textgrad_variable(self, var):
        """Helper method to extract value from TextGrad variables."""
        if hasattr(var, 'value'):
            return var.value
        return var
    
    def _hash_prompt(self, content: str, system_prompt: Optional[str] = None):
        """Create a hash for the prompt to use as cache key."""
        combined = f"{system_prompt if system_prompt else ''}|||{content}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def generate(self, content: str, system_prompt: Optional[str] = None, **kwargs):
        """Generate text using the LlamaCPP model."""
        content_text = self._process_textgrad_variable(content)
        system_prompt_text = self._process_textgrad_variable(system_prompt) if system_prompt else None
    
        # Check cache first if enabled
        if self.use_cache:
            cache_key = self._hash_prompt(content_text, system_prompt_text)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Prepare prompt
        if system_prompt_text:
            full_prompt = f"{system_prompt_text}\n\n{content_text}"
        else:
            full_prompt = content_text
        
        # Generate response using the LLM
        response = self.llm.complete(full_prompt)
        
        # Cache the result if enabled
        if self.use_cache:
            self.cache[cache_key] = response
        
        return response
    
    def __call__(self, content: str, system_prompt: Optional[str] = None, **kwargs):
        """Make the engine callable, same as generate."""
        return self.generate(content=content, system_prompt=system_prompt, **kwargs)
    
    def __str__(self):
        return self.name
    
    def backward(self, output, content=None, system_prompt=None, **kwargs):
        """Generate gradients for TextGrad optimization."""
        # Process the inputs
        if isinstance(output, str):
            output_text = output
        else:
            output_text = output.value if hasattr(output, 'value') else str(output)
        
        if content is None:
            return {"gradient": "This is a new improved documentation."}
        
        content_text = content.value if hasattr(content, 'value') else str(content)
        
        # Create a specific prompt for gradient generation
        gradient_prompt = f"""
        You are a documentation improvement expert. Your task is to REWRITE the API documentation 
        to directly address the issues mentioned in the evaluation.
        
        Current documentation:
        ```
        {content_text}
        ```
        
        Evaluation of current documentation:
        ```
        {output_text}
        ```
        
        IMPORTANT: Don't explain what changes to make. Instead, provide the COMPLETE REVISED DOCUMENTATION
        with all improvements already implemented. The output should be ONLY the updated documentation text
        that can directly replace the original.
        
        Generate the improved documentation now:
        """
        
        # Generate the improved documentation directly
        improved_doc = self.generate(content=gradient_prompt)
        
        # TextGrad expects a dict with a "gradient" key
        return {"gradient": improved_doc}


# ========================== OPTIMIZER CLASS ==========================

class APIDocOptimizer:
    """API Documentation Optimizer using TextGrad framework.
    
    This class implements an optimizer that takes API documentation as input
    and iteratively improves it using the TextGrad framework with LLMs.
    """
    
    def __init__(
        self, 
        model_name: str = "openai",
        model_path: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        use_local_model: bool = False,
        lmstudio_url: str = "http://localhost:1234/v1",
        lmstudio_model: Optional[str] = None,
        iterations: int = 3,
        temperature: float = 0.1,
        verbose: bool = True
    ):
        """Initialize the API Documentation Optimizer.
        
        Args:
            model_name: The name of the model to use ('peft', 'llama', 'openai', 'lmstudio')
            model_path: Path to local model (for peft or llama models)
            openai_api_key: OpenAI API key (for OpenAI models)
            use_local_model: Whether to use a local model with PEFT
            lmstudio_url: URL for LMStudio API server
            lmstudio_model: Model string for LMStudio
            iterations: Number of optimization iterations
            temperature: Temperature parameter for text generation
            verbose: Whether to print verbose output
        """
        self.model_name = model_name.lower()
        self.model_path = model_path
        self.openai_api_key = openai_api_key
        self.use_local_model = use_local_model
        self.lmstudio_url = lmstudio_url
        self.lmstudio_model = lmstudio_model
        self.iterations = iterations
        self.temperature = temperature
        self.verbose = verbose
        
        # Check model availability
        if self.model_name == "peft" and not PEFT_AVAILABLE:
            raise ImportError("PEFT and Transformers libraries are required. Install with: pip install peft transformers torch")
        elif self.model_name == "llama" and not LLAMACPP_AVAILABLE:
            raise ImportError("llama-index and llama-cpp-python are required. Install with: pip install llama-index llama-cpp-python")
        elif self.model_name == "openai" and not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library is required. Install with: pip install openai")
        elif self.model_name == "lmstudio" and not LMSTUDIO_AVAILABLE:
            raise ImportError("OpenAI and textgrad's local_model_openai_api module are required. Install with: pip install openai textgrad")
        
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
        if self.model_name == "peft" or self.use_local_model:
            try:
                if not self.model_path:
                    raise ValueError("model_path is required for PEFT model")
                
                self.engine = PeftBackwardEngine(
                    model_dir=self.model_path,
                    temperature=self.temperature,
                    use_cache=True
                )
                
                if self.verbose:
                    print(f"Initialized PEFT model from {self.model_path}")
                
            except ImportError:
                print("Error: Required packages for PEFT model not found.")
                print("Install with: pip install peft transformers torch")
                raise
        
        elif self.model_name == "lmstudio":
            try:
                self.engine = LMStudioEngine(
                    base_url=self.lmstudio_url,
                    api_key="lm-studio",
                    model_string=self.lmstudio_model,
                    temperature=self.temperature,
                    use_cache=True
                )
                
                if self.verbose:
                    print(f"Initialized LMStudio model at {self.lmstudio_url}")
                
            except ImportError:
                print("Error: Required packages for LMStudio not found.")
                print("Install with: pip install openai textgrad")
                raise
        
        elif self.model_name == "llama":
            try:
                if not self.model_path:
                    raise ValueError("model_path is required for Llama model")
                
                self.engine = LlamaCPPEngine(
                    model_path=self.model_path,
                    temperature=self.temperature,
                    use_cache=True
                )
                
                if self.verbose:
                    print(f"Initialized Llama model from {self.model_path}")
                
            except ImportError:
                print("Error: Required packages for Llama model not found.")
                print("Install with: pip install llama-index llama-cpp-python")
                raise
                
        elif self.model_name == "openai":
            # Use OpenAI models
            if not self.openai_api_key:
                self.openai_api_key = os.environ.get("OPENAI_API_KEY")
                if not self.openai_api_key:
                    raise ValueError("OpenAI API key is required for OpenAI models")
            
            # Set the API key
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
            
            # Initialize the OpenAI engine
            self.engine = get_engine("experimental:gpt-4o-mini", cache=False)
            
            if self.verbose:
                print("Initialized OpenAI model")
        
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
        # Set the backward engine for TextGrad
        tg.set_backward_engine(self.engine, override=True)
        
        if self.verbose:
            print(f"Using engine: {str(self.engine)}")
    
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
        loss_fn = tg.TextLoss(system_prompt, engine=self.engine)
        
        # Create the optimizer
        optimizer = tg.TGD([doc_var], engine=self.engine)
        
        # Store optimization history
        history = {
            "iteration": [0],
            "documentation": [documentation],
            "feedback": ["Initial version"]
        }
        
        # Optimization loop
        for i in range(self.iterations):
            if self.verbose:
                print(f"\nIteration {i+1}/{self.iterations}")
            
            try:
                # Compute the loss (evaluation)
                start_time = time.time()
                loss = loss_fn(doc_var)
                
                if self.verbose:
                    print(f"Evaluation feedback (took {time.time() - start_time:.2f}s):")
                    print(loss.value)
                    
                # Store feedback
                history["feedback"].append(loss.value)
                
                # Backward pass to compute gradients
                start_time = time.time()
                loss.backward()
                
                if self.verbose:
                    print(f"Backward pass took {time.time() - start_time:.2f}s")
                
                # Update the documentation
                if self.verbose:
                    print("Applying improvements...")
                optimizer.step()
                
                # Store the updated documentation
                history["documentation"].append(doc_var.value)
                history["iteration"].append(i+1)
                
                if self.verbose:
                    print(f"Updated documentation:\n{doc_var.value}")
                    
            except IndexError as e:
                # If optimizer fails, use manual update
                if self.verbose:
                    print(f"Optimizer error: {e}")
                    print("Falling back to manual update...")
                
                improved_doc = self.manual_update(doc_var.value, loss.value if 'loss' in locals() else "Improve this documentation for clarity and completeness.")
                doc_var.value = improved_doc
                
                # Store the updated documentation
                history["documentation"].append(doc_var.value)
                history["iteration"].append(i+1)
                if 'loss' not in locals():
                    history["feedback"].append("Error during evaluation")
                
                if self.verbose:
                    print(f"Updated documentation (manual fallback):\n{doc_var.value}")
            
            except Exception as e:
                if self.verbose:
                    print(f"Unexpected error during optimization: {e}")
                    print("Falling back to manual update...")
                
                improved_doc = self.manual_update(doc_var.value, "Improve this documentation for clarity and completeness.")
                doc_var.value = improved_doc
                
                # Store the updated documentation
                history["documentation"].append(doc_var.value)
                history["iteration"].append(i+1)
                history["feedback"].append(f"Error: {str(e)}")
                
                if self.verbose:
                    print(f"Updated documentation (error fallback):\n{doc_var.value}")
        
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
        
        # Get the evaluation response
        try:
            response = self.engine.generate(
                content="Please evaluate the documentation according to the criteria provided.",
                system_prompt=eval_prompt
            )
            
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
                    if self.verbose:
                        print("Warning: Could not find JSON in evaluation response.")
                        print(f"Response: {response}")
                    
                    scores = {
                        "completeness": 5,
                        "accuracy": 5,
                        "usability": 5,
                        "overall": 5,
                        "feedback": "Failed to parse evaluation: No JSON found in response."
                    }
            except json.JSONDecodeError as e:
                if self.verbose:
                    print(f"Warning: JSON decode error: {e}")
                    print(f"Response: {response}")
                
                scores = {
                    "completeness": 5,
                    "accuracy": 5,
                    "usability": 5, 
                    "overall": 5,
                    "feedback": f"Failed to parse evaluation: Invalid JSON format. Error: {str(e)}"
                }
        except Exception as e:
            if self.verbose:
                print(f"Error during evaluation: {str(e)}")
            
            scores = {
                "completeness": 5,
                "accuracy": 5,
                "usability": 5, 
                "overall": 5,
                "feedback": f"Error during evaluation: {str(e)}"
            }
        
        return scores
    
    def manual_update(self, documentation, feedback):
        """Manually update documentation based on feedback when optimizer fails."""
        # Create an explicit prompt for the model to generate improved documentation
        update_prompt = f"""
        Please improve this documentation based on the feedback:

        CURRENT DOCUMENTATION:
        ```
        {documentation}
        ```

        FEEDBACK:
        {feedback}

        IMPORTANT: Provide ONLY the complete improved documentation text, without explanations.
        """
        
        try:
            improved_doc = self.engine.generate(content=update_prompt)
            
            # Clean up common formatting issues
            improved_doc = improved_doc.replace("```", "").strip()
            if improved_doc.lower().startswith("documentation:"):
                improved_doc = improved_doc[13:].strip()
                
            return improved_doc
        
        except Exception as e:
            if self.verbose:
                print(f"Error during manual update: {str(e)}")
                print("Returning original documentation.")
            
            return documentation

# ========================== MAIN FUNCTION ==========================

def main():
    """Main function to demonstrate the API Documentation Optimizer."""
    parser = argparse.ArgumentParser(description="API Documentation Optimizer")
    parser.add_argument("--input", type=str, help="Input documentation file or string")
    parser.add_argument("--model", type=str, default="openai", choices=["openai", "peft", "llama", "lmstudio"], 
                      help="Model type to use (openai, peft, llama, or lmstudio)")
    parser.add_argument("--model-path", type=str, help="Path to local model (for peft or llama models)")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (for OpenAI models)")
    parser.add_argument("--local", action="store_true", help="Use local model with PEFT")
    parser.add_argument("--lmstudio-url", type=str, default="http://localhost:1234/v1", 
                      help="URL for LMStudio API server")
    parser.add_argument("--lmstudio-model", type=str, 
                      help="Model string for LMStudio (e.g., 'Llama-3.2-1B-Instruct-Q8_0-GGUF/llama-3.2-1b-instruct-q8_0.gguf')")
    parser.add_argument("--iterations", type=int, default=3, help="Number of optimization iterations")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for text generation")
    parser.add_argument("--output", type=str, help="Output file for optimized documentation")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
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
    
    try:
        # Create the optimizer
        optimizer = APIDocOptimizer(
            model_name=args.model,
            model_path=args.model_path,
            openai_api_key=args.api_key,
            use_local_model=args.local,
            lmstudio_url=args.lmstudio_url,
            lmstudio_model=args.lmstudio_model,
            iterations=args.iterations,
            temperature=args.temperature,
            verbose=args.verbose
        )
        
        # Evaluate initial documentation
        print("Evaluating initial documentation...")
        initial_scores = optimizer.evaluate_documentation(documentation)
        print(f"Initial scores: {initial_scores}")
        
        # Optimize the documentation with fallback mechanism
        print("\nOptimizing documentation...")
        try:
            # Try using the TextGrad optimizer
            optimized_doc = optimizer.optimize(documentation)
        except IndexError as e:
            # If optimizer fails with IndexError (common with TextGrad), use manual update
            print(f"Optimizer error: {e}")
            print("Falling back to manual update...")
            feedback = initial_scores.get('feedback', 'Improve this documentation for clarity and completeness.')
            optimized_doc = optimizer.manual_update(documentation, feedback)
        except Exception as e:
            # Catch any other exceptions
            print(f"Unexpected error during optimization: {e}")
            print("Falling back to manual update...")
            feedback = initial_scores.get('feedback', 'Improve this documentation for clarity and completeness.')
            optimized_doc = optimizer.manual_update(documentation, feedback)
        
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
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as e:
        print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()