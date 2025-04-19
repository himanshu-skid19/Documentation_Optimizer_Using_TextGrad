import streamlit as st
import json
import textgrad as tg
from textgrad.engine_experimental.openai import OpenAIEngine
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from textgrad.engine import get_engine
import hashlib
import diskcache as dc
from typing import Dict, List, Any, Optional, Tuple

# Try importing modules with proper error handling
try:
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from textgrad.engine.local_model_openai_api import ChatExternalClient
    LMSTUDIO_AVAILABLE = True
except ImportError:
    LMSTUDIO_AVAILABLE = False

try:
    from llama_index.llms import LlamaCPP
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False

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
        """Initialize the PEFT engine."""
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
        return {"gradient": improved_doc}


class LlamaCPPEngine:
    """TextGrad engine implementation for LlamaCPP models."""
    
    def __init__(self, model_path, temperature=0.1, use_cache=True, cache_dir=".textgrad_cache"):
        """Initialize the LlamaCPP engine."""
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


class APIDocOptimizerApp:
    """Streamlit application for API Documentation Optimization using TextGrad."""
    
    def __init__(self):
        """Initialize the application."""
        self.setup_page()
        self.engine = None
        self.eval_engine = None  # Separate engine for evaluation
        self.load_engine()
        self.optimization_history = []
        
    def setup_page(self):
        """Set up the Streamlit page configuration."""
        st.set_page_config(
            page_title="API Documentation Optimizer",
            page_icon="📝",
            layout="wide"
        )
        st.title("API Documentation Optimizer")
        st.sidebar.header("Configuration")
    
    def load_engine(self):
        """Load the TextGrad engine with the appropriate LLM."""
        # Always ask for OpenAI API key for evaluation
        st.sidebar.subheader("OpenAI API Key (Required for Evaluation)")
        api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        
        if not api_key:
            st.sidebar.warning("⚠️ OpenAI API key is required for evaluation")
        else:
            os.environ["OPENAI_API_KEY"] = api_key
            try:
                # Initialize evaluation engine with GPT-4o-mini
                self.eval_engine = get_engine("experimental:gpt-4o-mini", cache=False)
                st.sidebar.success("✅ OpenAI GPT-4o-mini loaded for evaluation!")
            except Exception as e:
                st.sidebar.error(f"Error loading OpenAI model: {str(e)}")
                self.eval_engine = None
        
        # Add model selection for optimization in sidebar
        st.sidebar.subheader("Model for Optimization")
        model_type = st.sidebar.selectbox(
            "Select Model",
            ["OpenAI", "Local Model (LMStudio)", "PEFT", "LlamaCPP"]
        )
        
        # Set default engine to None
        self.engine = None
        
        if model_type == "OpenAI":
            if not OPENAI_AVAILABLE:
                st.sidebar.error("OpenAI library not found. Install with: pip install openai")
                return
                
            # Already have API key from above
            model_name = st.sidebar.selectbox(
                "Model", 
                ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                index=0
            )
            
            if api_key:
                try:
                    self.engine = get_engine(f"experimental:{model_name}", cache=False)
                    st.sidebar.success(f"OpenAI {model_name} loaded for optimization!")
                except Exception as e:
                    st.sidebar.error(f"Error loading OpenAI model: {str(e)}")
            else:
                st.sidebar.warning("Please enter your OpenAI API key")
                
        elif model_type == "Local Model (LMStudio)":
            if not LMSTUDIO_AVAILABLE:
                st.sidebar.error(
                    "Required packages for LMStudio not found. "
                    "Install with: pip install openai textgrad"
                )
                return
                
            st.sidebar.info(
                "Make sure LMStudio is running with a local server at http://localhost:1234/v1"
            )
            
            base_url = st.sidebar.text_input("LMStudio URL", value="http://localhost:1234/v1")
            model_name = st.sidebar.text_input(
                "Model Name", 
                value="Llama-3.2-1B-Instruct-Q8_0-GGUF"
            )
            temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
            
            try:
                client = OpenAI(base_url=base_url, api_key="lm-studio")
                self.engine = ChatExternalClient(client=client, model_string=model_name)
                st.sidebar.success(f"Local model loaded for optimization: {model_name}")
            except Exception as e:
                st.sidebar.error(f"Error loading LMStudio model: {str(e)}")
                
        elif model_type == "PEFT":
            if not PEFT_AVAILABLE:
                st.sidebar.error(
                    "Required packages for PEFT not found. "
                    "Install with: pip install peft transformers torch"
                )
                return
                
            model_path = st.sidebar.text_input("PEFT Model Directory", value="api-docs-model")
            temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
            use_cache = st.sidebar.checkbox("Use Cache", value=True)
            
            if not os.path.exists(model_path):
                st.sidebar.warning(f"Model path not found: {model_path}")
                return
                
            try:
                self.engine = PeftBackwardEngine(
                    model_dir=model_path,
                    temperature=temperature,
                    use_cache=use_cache
                )
                st.sidebar.success(f"PEFT model loaded for optimization from: {model_path}")
            except Exception as e:
                st.sidebar.error(f"Error loading PEFT model: {str(e)}")
                
        elif model_type == "LlamaCPP":
            if not LLAMACPP_AVAILABLE:
                st.sidebar.error(
                    "Required packages for LlamaCPP not found. "
                    "Install with: pip install llama-index llama-cpp-python"
                )
                return
                
            model_path = st.sidebar.text_input("LlamaCPP Model Path", value="")
            temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
            use_cache = st.sidebar.checkbox("Use Cache", value=True)
            
            if not os.path.exists(model_path):
                st.sidebar.warning(f"Model path not found: {model_path}")
                return
                
            try:
                self.engine = LlamaCPPEngine(
                    model_path=model_path,
                    temperature=temperature,
                    use_cache=use_cache
                )
                st.sidebar.success(f"LlamaCPP model loaded for optimization from: {model_path}")
            except Exception as e:
                st.sidebar.error(f"Error loading LlamaCPP model: {str(e)}")
        
        # Set the backward engine for TextGrad if engine is loaded
        if self.engine:
            tg.set_backward_engine(self.engine, override=True)
            
        # Display model information
        st.sidebar.subheader("Active Models")
        eval_model_info = "GPT-4o-mini (OpenAI)" if self.eval_engine else "Not loaded"
        st.sidebar.info(f"Evaluation: {eval_model_info}")
        
        opt_model_info = str(self.engine) if self.engine else "Not loaded"
        st.sidebar.info(f"Optimization: {opt_model_info}")
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the documentation evaluation."""
        return """You are an expert API documentation reviewer specialized in optimizing technical documentation.
        
        Evaluate the given API documentation for the following criteria:
        1. Completeness: Does it cover all essential elements (description, parameters, return values, examples)?
        2. Technical Accuracy: Is the information correct and precise?
        3. Usability: Is the documentation clear, practical, and helpful for developers?
        
        Identify specific problems in the documentation and provide detailed, constructive feedback on how to improve it.
        Focus on making the documentation more helpful for developers who will use the API.
        
        Be specific about what changes should be made to improve the documentation quality.
        """
    
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
        ```
        {feedback}
        ```

        IMPORTANT: Provide ONLY the complete improved documentation text, without explanations.
        The documentation should follow proper formatting with clear sections for Description,
        Parameters, Returns, and Examples.
        """
        
        try:
            improved_doc = self.engine.generate(content=update_prompt)
            
            # Clean up common formatting issues
            improved_doc = improved_doc.replace("```", "").strip()
            if improved_doc.lower().startswith("documentation:"):
                improved_doc = improved_doc[13:].strip()
                
            return improved_doc
        
        except Exception as e:
            st.error(f"Error during manual update: {str(e)}")
            st.warning("Returning original documentation.")
            
            return documentation
    
    def optimize_documentation(self, documentation: str, iterations: int = 1) -> str:
        """Optimize the given API documentation using TextGrad."""
        if not self.engine:
            st.error("No optimization engine loaded. Please configure a model first.")
            return documentation
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
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
        
        # Feedback container
        feedback_container = st.expander("Optimization Process", expanded=True)
        
        # Optimization loop
        for i in range(iterations):
            iteration_progress = (i / iterations)
            progress_bar.progress(iteration_progress)
            status_text.text(f"Optimization iteration {i+1}/{iterations}")
            
            with feedback_container:
                st.subheader(f"Iteration {i+1}")
                
                # Compute the loss (evaluation)
                with st.spinner("Evaluating documentation..."):
                    try:
                        start_time = time.time()
                        loss = loss_fn(doc_var)
                        eval_time = time.time() - start_time
                        st.text_area(
                            f"Evaluation Feedback (took {eval_time:.2f}s)", 
                            value=loss.value,
                            height=150
                        )
                        feedback = loss.value
                    except Exception as e:
                        st.error(f"Error during evaluation: {str(e)}")
                        feedback = "Error during evaluation. Proceeding with manual update."
                        continue
                
                # Backward pass to compute gradients & Update the documentation
                with st.spinner("Optimizing documentation..."):
                    try:
                        # Backward pass
                        start_time = time.time()
                        loss.backward()
                        backward_time = time.time() - start_time
                        
                        # Update documentation
                        start_time = time.time()
                        optimizer.step()
                        step_time = time.time() - start_time
                        
                        st.text(f"Backward pass: {backward_time:.2f}s, Update: {step_time:.2f}s")
                        
                    except IndexError as e:
                        # If optimizer fails with IndexError, use manual update
                        st.warning(f"Optimizer error: {str(e)}")
                        st.info("Falling back to manual update...")
                        improved_doc = self.manual_update(doc_var.value, feedback)
                        doc_var.value = improved_doc
                    except Exception as e:
                        # Catch any other exceptions
                        st.error(f"Unexpected error during optimization: {str(e)}")
                        st.info("Falling back to manual update...")
                        improved_doc = self.manual_update(doc_var.value, feedback)
                        doc_var.value = improved_doc
                
                # Display improved documentation
                st.text_area(
                    "Improved Documentation", 
                    value=doc_var.value,
                    height=200
                )
            
            # Add to optimization history
            self.optimization_history.append({
                "iteration": i + 1,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "feedback": feedback if 'feedback' in locals() else "No feedback available",
                "documentation": doc_var.value
            })
        
        # Complete the progress bar
        progress_bar.progress(1.0)
        status_text.text("Optimization complete!")
        
        return doc_var.value
    
    def evaluate_documentation(self, documentation: str) -> dict:
        """Evaluate the API documentation using GPT-4o-mini and return metrics."""
        if not self.eval_engine:
            st.error("OpenAI evaluation engine not loaded. Please provide an API key.")
            return {
                "completeness": 0,
                "accuracy": 0,
                "usability": 0,
                "overall": 0,
                "feedback": "OpenAI API key is required for evaluation"
            }
        
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
        
        with st.spinner("Evaluating documentation with GPT-4o-mini..."):
            try:
                # Get the evaluation response using specifically the OpenAI evaluation engine
                response = self.eval_engine.generate(
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
                        st.warning("Could not extract JSON from evaluation response.")
                        scores = {
                            "completeness": 5,
                            "accuracy": 5,
                            "usability": 5,
                            "overall": 5,
                            "feedback": response
                        }
                except json.JSONDecodeError:
                    st.warning("Failed to parse JSON in evaluation response.")
                    scores = {
                        "completeness": 5,
                        "accuracy": 5,
                        "usability": 5, 
                        "overall": 5,
                        "feedback": response
                    }
            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")
                scores = {
                    "completeness": 0,
                    "accuracy": 0,
                    "usability": 0, 
                    "overall": 0,
                    "feedback": f"Error during evaluation: {str(e)}"
                }
        
        return scores
    
    def display_optimization_history(self):
        """Display the optimization history."""
        if not self.optimization_history:
            st.info("No optimization history yet.")
            return
        
        st.subheader("Optimization History")
        
        for i, entry in enumerate(self.optimization_history):
            with st.expander(f"Iteration {entry['iteration']} - {entry['timestamp']}"):
                st.text_area("Feedback", value=entry['feedback'], height=100)
                st.text_area("Documentation", value=entry['documentation'], height=150)
    
    def run(self):
        """Run the Streamlit application."""
        # Sidebar configuration
        st.sidebar.header("Optimization Settings")
        iterations = st.sidebar.slider("Optimization Iterations", 1, 5, 2)
        
        # Main interface
        st.header("Current Documentation")
        
        # Example documentation template
        example_doc = """Description
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
        
        # Documentation input
        current_doc = st.text_area(
            "Enter API Documentation",
            value=example_doc,
            height=300
        )
        
        # Actions in columns
        col1, col2 = st.columns(2)
        
        # Initial evaluation
        with col1:
            if st.button("Evaluate Documentation", use_container_width=True):
                if current_doc:
                    if not self.eval_engine:
                        st.error("Please enter an OpenAI API key for evaluation.")
                    else:
                        evaluation = self.evaluate_documentation(current_doc)
                        
                        # Display evaluation metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Completeness", f"{evaluation['completeness']}/10")
                        col2.metric("Technical Accuracy", f"{evaluation['accuracy']}/10")
                        col3.metric("Usability", f"{evaluation['usability']}/10")
                        col4.metric("Overall", f"{evaluation['overall']}/10")
                        
                        # Display detailed feedback
                        st.text_area(
                            "Detailed Feedback",
                            value=evaluation['feedback'],
                            height=150
                        )
                else:
                    st.warning("Please enter documentation to evaluate.")
        
        # Optimization
        with col2:
            if st.button("Start Optimization", use_container_width=True):
                if current_doc:
                    if not self.engine:
                        st.error("Please configure an optimization model first.")
                    elif not self.eval_engine:
                        st.error("Please enter an OpenAI API key for evaluation.")
                    else:
                        try:
                            optimized_doc = self.optimize_documentation(
                                current_doc,
                                iterations=iterations
                            )
                            
                            # Display the optimized documentation
                            st.subheader("Optimized Documentation")
                            st.text_area(
                                "Final Result",
                                value=optimized_doc,
                                height=300
                            )
                            
                            # Allow copying to clipboard
                            if st.button("Copy to Clipboard"):
                                st.code(optimized_doc)
                                st.success("Copied to clipboard! (Use the copy button in the code block)")
                            
                            # Evaluate the optimized documentation
                            with st.expander("Final Evaluation", expanded=True):
                                final_evaluation = self.evaluate_documentation(optimized_doc)
                                
                                # Display comparison of metrics
                                initial_evaluation = self.evaluate_documentation(current_doc)
                                
                                # Display metrics in columns
                                cols = st.columns(4)
                                for i, metric in enumerate(["completeness", "accuracy", "usability", "overall"]):
                                    metric_name = metric.capitalize()
                                    initial_val = initial_evaluation.get(metric, 0)
                                    final_val = final_evaluation.get(metric, 0)
                                    diff = final_val - initial_val
                                    
                                    cols[i].metric(
                                        metric_name, 
                                        f"{final_val}/10",
                                        f"{diff:+.1f}"
                                    )
                                
                                # Display detailed feedback
                                st.text_area(
                                    "Detailed Feedback",
                                    value=final_evaluation['feedback'],
                                    height=150
                                )
                                
                        except Exception as e:
                            st.error(f"Error during optimization: {str(e)}")
                else:
                    st.warning("Please enter documentation to optimize.")
        
        # Reset button
        if st.sidebar.button("Reset History"):
            self.optimization_history = []
            st.experimental_rerun()
        
        # Display optimization history
        st.header("Optimization History")
        self.display_optimization_history()


if __name__ == "__main__":
    app = APIDocOptimizerApp()
    app.run()