import streamlit as st
import json
import textgrad as tg
from textgrad.engine_experimental.openai import OpenAIEngine
import os
from datetime import datetime
from dotenv import load_dotenv
from textgrad.engine import get_engine
load_dotenv()

class APIDocOptimizerApp:
    """Streamlit application for API Documentation Optimization using TextGrad."""
    
    def __init__(self):
        """Initialize the application."""
        self.setup_page()
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
        # Add model selection in sidebar
        model_type = st.sidebar.selectbox(
            "Select Model",
            ["OpenAI", "Local Model (LMStudio)"]
        )
        
        if model_type == "OpenAI":
            api_key = st.sidebar.text_input("OpenAI API Key", type="password")
            
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                self.engine = get_engine("experimental:gpt-4o-mini", cache=False)
                st.sidebar.success("OpenAI model loaded!")
            else:
                st.sidebar.warning("Please enter your OpenAI API key")
                self.engine = None
                
        elif model_type == "Local Model (LMStudio)":
            st.sidebar.info(
                "Make sure LMStudio is running with a local server at http://localhost:1234/v1"
            )
            
            try:
                from openai import OpenAI
                from textgrad.engine.local_model_openai_api import ChatExternalClient
                
                # Use default LMStudio address
                client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
                
                # Allow model selection
                model_name = st.sidebar.text_input(
                    "Model Name", 
                    value="mlabonne/NeuralBeagle14-7B-GGUF"
                )
                
                self.engine = ChatExternalClient(client=client, model_string=model_name)
                st.sidebar.success(f"Local model loaded: {model_name}")
                
            except ImportError:
                st.sidebar.error(
                    "Required packages for local model not found. "
                    "Install with: pip install openai textgrad"
                )
                self.engine = None
                
        # Set the backward engine for TextGrad if engine is loaded
        if self.engine:
            tg.set_backward_engine(self.engine, override=True)
    
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
    
    def optimize_documentation(self, documentation: str, iterations: int = 1) -> str:
        """Optimize the given API documentation using TextGrad.
        
        Args:
            documentation: The API documentation to optimize
            iterations: Number of optimization iterations
            
        Returns:
            The optimized API documentation
        """
        if not self.engine:
            st.error("No TextGrad engine loaded. Please configure a model first.")
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
        loss_fn = tg.TextLoss(system_prompt)
        
        # Create the optimizer
        optimizer = tg.TGD([doc_var])
        
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
                    loss = loss_fn(doc_var)
                    st.text_area(
                        "Evaluation Feedback", 
                        value=loss.value,
                        height=150
                    )
                
                # Backward pass to compute gradients
                with st.spinner("Computing improvements..."):
                    loss.backward()
                
                # Update the documentation
                with st.spinner("Applying improvements..."):
                    optimizer.step()
                    st.text_area(
                        "Improved Documentation", 
                        value=doc_var.value,
                        height=200
                    )
            
            # Add to optimization history
            self.optimization_history.append({
                "iteration": i + 1,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "feedback": loss.value,
                "documentation": doc_var.value
            })
        
        # Complete the progress bar
        progress_bar.progress(1.0)
        status_text.text("Optimization complete!")
        
        return doc_var.value
    
    def evaluate_documentation(self, documentation: str) -> dict:
        """Evaluate the API documentation and return metrics.
        
        Args:
            documentation: The API documentation to evaluate
            
        Returns:
            Dictionary with evaluation scores and feedback
        """
        if not self.engine:
            st.error("No TextGrad engine loaded. Please configure a model first.")
            return {
                "completeness": 0,
                "accuracy": 0,
                "usability": 0,
                "overall": 0,
                "feedback": "No engine loaded"
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
        
        # Create a TextGrad variable for the documentation to evaluate
        eval_var = tg.Variable(
            eval_prompt,
            requires_grad=False,
            role_description="evaluation request"
        )
        
        with st.spinner("Evaluating documentation..."):
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
        
        # Initial evaluation
        if st.button("Evaluate Documentation"):
            if current_doc:
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
        st.header("Documentation Optimization")
        
        if st.button("Start Optimization"):
            if current_doc:
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
                
                # Evaluate the optimized documentation
                st.subheader("Final Evaluation")
                final_evaluation = self.evaluate_documentation(optimized_doc)
                
                # Display final evaluation metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(
                    "Completeness", 
                    f"{final_evaluation['completeness']}/10"
                )
                col2.metric(
                    "Technical Accuracy", 
                    f"{final_evaluation['accuracy']}/10"
                )
                col3.metric(
                    "Usability", 
                    f"{final_evaluation['usability']}/10"
                )
                col4.metric(
                    "Overall", 
                    f"{final_evaluation['overall']}/10"
                )
            else:
                st.warning("Please enter documentation to optimize.")
        
        # Reset button
        if st.sidebar.button("Reset"):
            self.optimization_history = []
            st.experimental_rerun()
        
        # Display optimization history
        st.header("Optimization History")
        self.display_optimization_history()


if __name__ == "__main__":
    app = APIDocOptimizerApp()
    app.run()