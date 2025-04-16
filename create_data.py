import os
import json
import yaml
import requests
import re
from pathlib import Path
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import random
import shutil
import subprocess
from bs4 import BeautifulSoup
import tempfile

def download_github_rest_api_spec():
    """Download GitHub's REST API OpenAPI specification."""
    print("Downloading GitHub REST API specification...")
    
    # Create directory
    os.makedirs("github_api", exist_ok=True)
    
    # GitHub provides their OpenAPI spec in this repository
    spec_url = "https://raw.githubusercontent.com/github/rest-api-description/main/descriptions/api.github.com/api.github.com.json"
    
    response = requests.get(spec_url)
    if response.status_code == 200:
        spec_path = os.path.join("github_api", "github_rest_api.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"GitHub REST API specification downloaded to {spec_path}")
        return spec_path
    else:
        print(f"Failed to download GitHub API spec: {response.status_code}")
        return None

def clone_kaggle_api_repo():
    """Clone the Kaggle API repository."""
    print("Cloning Kaggle API repository...")
    
    # Clone the repository
    repo_url = "https://github.com/Kaggle/kaggle-api.git"
    repo_path = "kaggle_api"
    
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
    
    try:
        subprocess.run(["git", "clone", repo_url, repo_path], check=True)
        print(f"Kaggle API repository cloned to {repo_path}")
        return repo_path
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone Kaggle API repository: {e}")
        return None

def extract_kaggle_api_docs(repo_path):
    """Extract API documentation from Kaggle API Python client."""
    if not repo_path or not os.path.exists(repo_path):
        return []
    
    print("Extracting Kaggle API documentation...")
    
    endpoints = []
    api_client_path = os.path.join(repo_path, "kaggle", "api", "kaggle_api_extended.py")
    
    if not os.path.exists(api_client_path):
        print(f"Kaggle API client file not found at {api_client_path}")
        return endpoints
    
    # Parse Python file to extract method documentation
    with open(api_client_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find class methods with docstrings
    class_pattern = r"class\s+(\w+)\(.*?\):"
    method_pattern = r"def\s+(\w+)\(self,\s*(.*?)\):\s*(?:(?:'''|\"\"\")(.*?)(?:'''|\"\"\"))??"
    
    for class_match in re.finditer(class_pattern, content, re.DOTALL):
        class_name = class_match.group(1)
        class_pos = class_match.end()
        
        # Find methods in this class
        for method_match in re.finditer(method_pattern, content[class_pos:], re.DOTALL):
            method_name = method_match.group(1)
            parameters_str = method_match.group(2)
            docstring = method_match.group(3) if method_match.group(3) else ""
            
            # Skip private methods
            if method_name.startswith("_"):
                continue
            
            # Extract method description from docstring
            description = docstring.strip() if docstring else f"{method_name} method"
            
            # Extract parameters
            parameters = []
            if parameters_str:
                param_list = parameters_str.split(",")
                for param in param_list:
                    param = param.strip()
                    if not param or param == "":
                        continue
                    
                    # Extract parameter name and default value
                    param_parts = param.split("=")
                    param_name = param_parts[0].strip()
                    param_required = len(param_parts) == 1  # Required if no default value
                    
                    # Try to extract parameter description from docstring
                    param_desc = ""
                    param_pattern = rf"{re.escape(param_name)}:.*?(?:\n|\r\n)"
                    param_match = re.search(param_pattern, docstring, re.IGNORECASE)
                    if param_match:
                        param_desc = param_match.group(0).strip()
                    
                    parameters.append({
                        "name": param_name,
                        "in": "query",  # Assume query parameters for API
                        "required": param_required,
                        "description": param_desc,
                        "type": "string"  # Default type
                    })
            
            # Create endpoint entry
            endpoint = {
                "path": f"/kaggle/{class_name.lower()}/{method_name}",
                "method": "GET",  # Default method
                "summary": f"{class_name}.{method_name}",
                "description": description,
                "parameters": parameters,
                "responses": {
                    "200": {
                        "description": "Successful operation"
                    }
                },
                "source": "Kaggle API"
            }
            
            endpoints.append(endpoint)
    
    print(f"Extracted {len(endpoints)} endpoints from Kaggle API")
    return endpoints

def process_openapi_spec(file_path):
    """Extract API endpoints from an OpenAPI specification."""
    print(f"Processing OpenAPI spec: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.json'):
            spec = json.load(f)
        else:
            spec = yaml.safe_load(f)
    
    endpoints = []
    
    # Handle different OpenAPI versions
    if 'swagger' in spec:  # OpenAPI 2.0
        base_path = spec.get('basePath', '')
        paths = spec.get('paths', {})
    else:  # OpenAPI 3.0+
        base_path = ''
        paths = spec.get('paths', {})
    
    # Process each path and method
    for path, methods in paths.items():
        for method, details in methods.items():
            if method in ['get', 'post', 'put', 'delete', 'patch', 'options', 'head']:
                # Extract parameters
                parameters = []
                for param in details.get('parameters', []):
                    param_info = {
                        'name': param.get('name', ''),
                        'in': param.get('in', ''),
                        'required': param.get('required', False),
                        'description': param.get('description', ''),
                        'type': get_parameter_type(param, spec),
                    }
                    parameters.append(param_info)
                
                # Extract responses
                responses = {}
                for status_code, response in details.get('responses', {}).items():
                    response_info = {
                        'description': response.get('description', ''),
                        'content': {}
                    }
                    
                    # OpenAPI 3.0+
                    if 'content' in response:
                        for media_type, media_info in response['content'].items():
                            if 'example' in media_info:
                                example = media_info['example']
                                if isinstance(example, (dict, list)):
                                    try:
                                        example = json.dumps(example, indent=2)
                                    except:
                                        example = str(example)
                                response_info['content'][media_type] = {
                                    'example': example
                                }
                    
                    responses[status_code] = response_info
                
                endpoint = {
                    'path': f"{base_path}{path}",
                    'method': method.upper(),
                    'summary': details.get('summary', ''),
                    'description': details.get('description', ''),
                    'parameters': parameters,
                    'responses': responses,
                    'source': os.path.basename(file_path)
                }
                
                endpoints.append(endpoint)
    
    print(f"Extracted {len(endpoints)} endpoints from {file_path}")
    return endpoints

def get_parameter_type(param, spec):
    """Get the type of a parameter."""
    # OpenAPI 3.0+
    if 'schema' in param:
        schema = param['schema']
        return get_type_from_schema(schema)
    
    # OpenAPI 2.0
    return param.get('type', '')

def get_type_from_schema(schema):
    """Extract type information from a schema object."""
    if not schema:
        return ''
        
    type_value = schema.get('type', '')
    
    if type_value == 'array' and 'items' in schema:
        items_type = get_type_from_schema(schema['items'])
        return f"array of {items_type}"
    
    format_value = schema.get('format', '')
    if format_value:
        return f"{type_value} ({format_value})"
        
    return type_value

def format_full_documentation(endpoint):
    """Format complete documentation for an endpoint."""
    doc = f"# {endpoint['method']} {endpoint['path']}\n\n"
    
    if endpoint['summary']:
        doc += f"{endpoint['summary']}\n\n"
    if endpoint['description']:
        doc += f"{endpoint['description']}\n\n"
    
    # Parameters
    if endpoint['parameters']:
        doc += "## Parameters\n\n"
        
        for param in endpoint['parameters']:
            required = " (Required)" if param.get('required', False) else " (Optional)"
            location = f" ({param.get('in', '')})" if param.get('in') else ""
            type_info = f" - {param.get('type', '')}" if param.get('type') else ""
            
            doc += f"- `{param.get('name', '')}`{required}{location}{type_info}: {param.get('description', '')}\n"
        
        doc += "\n"
    
    # Responses
    if endpoint['responses']:
        doc += "## Responses\n\n"
        
        for status_code, response_info in endpoint['responses'].items():
            doc += f"### {status_code}\n\n"
            
            if response_info.get('description'):
                doc += f"{response_info['description']}\n\n"
            
            if response_info.get('content', {}).get('application/json', {}).get('example'):
                doc += "```json\n"
                doc += f"{response_info['content']['application/json']['example']}\n"
                doc += "```\n\n"
    
    # Usage example
    doc += "## Example\n\n"
    
    # Create a curl example
    curl_example = create_curl_example(endpoint)
    doc += f"```bash\n{curl_example}\n```\n\n"
    
    # Create a Python example
    python_example = create_python_example(endpoint)
    doc += f"```python\n{python_example}\n```\n\n"
    
    return doc.strip()

def format_parameters_documentation(endpoint):
    """Format parameter documentation for an endpoint."""
    doc = "## Parameters\n\n"
    
    if not endpoint['parameters']:
        doc += "This endpoint does not require any parameters.\n"
        return doc
    
    # Group parameters by location
    params_by_location = {}
    for param in endpoint['parameters']:
        location = param.get('in', 'other')
        if location not in params_by_location:
            params_by_location[location] = []
        params_by_location[location].append(param)
    
    # Document parameters by location
    for location, params in params_by_location.items():
        if location != 'other':
            doc += f"### {location.capitalize()} Parameters\n\n"
        
        for param in params:
            required = " (Required)" if param.get('required', False) else " (Optional)"
            type_info = f" - {param.get('type', '')}" if param.get('type') else ""
            
            doc += f"- `{param.get('name', '')}`{required}{type_info}: {param.get('description', '')}\n"
        
        doc += "\n"
    
    return doc.strip()

def format_response_documentation(endpoint):
    """Format response documentation for an endpoint."""
    doc = "## Responses\n\n"
    
    if not endpoint['responses']:
        doc += "Response documentation not available.\n"
        return doc
    
    for status_code, response_info in endpoint['responses'].items():
        doc += f"### {status_code}\n\n"
        
        if response_info.get('description'):
            doc += f"{response_info['description']}\n\n"
        
        if response_info.get('content', {}).get('application/json', {}).get('example'):
            doc += "```json\n"
            doc += f"{response_info['content']['application/json']['example']}\n"
            doc += "```\n\n"
    
    return doc.strip()

def create_curl_example(endpoint):
    """Create a cURL example for an endpoint."""
    path = endpoint['path']
    method = endpoint['method']
    
    # Replace path parameters with values
    for param in endpoint['parameters']:
        if param.get('in') == 'path':
            param_name = param.get('name', '')
            path = path.replace(f"{{{param_name}}}", "123")
    
    # Use the appropriate base URL
    if "github" in endpoint.get('source', '').lower() or "github" in path.lower():
        base_url = "https://api.github.com"
    elif "kaggle" in endpoint.get('source', '').lower() or "kaggle" in path.lower():
        base_url = "https://www.kaggle.com/api/v1"
    else:
        base_url = "https://api.example.com"
    
    curl = f"curl -X {method} {base_url}{path}"
    
    # Add headers
    curl += ' \\\n  -H "Content-Type: application/json"'
    
    if "github" in endpoint.get('source', '').lower() or "github" in path.lower():
        curl += ' \\\n  -H "Authorization: token YOUR_GITHUB_TOKEN"'
    elif "kaggle" in endpoint.get('source', '').lower() or "kaggle" in path.lower():
        curl += ' \\\n  -H "Authorization: Basic YOUR_KAGGLE_API_KEY"'
    else:
        curl += ' \\\n  -H "Authorization: Bearer YOUR_API_KEY"'
    
    # Add query parameters
    query_params = []
    for param in endpoint['parameters']:
        if param.get('in') == 'query':
            param_name = param.get('name', '')
            param_type = param.get('type', '').lower()
            
            if param_type == 'boolean':
                query_params.append(f"{param_name}=true")
            elif param_type == 'integer' or param_type == 'number':
                query_params.append(f"{param_name}=123")
            else:
                query_params.append(f"{param_name}=example")
    
    if query_params and method in ['GET', 'DELETE']:
        curl += f"?{'&'.join(query_params)}"
    
    # Add request body
    if method in ['POST', 'PUT', 'PATCH']:
        body_params = {}
        for param in endpoint['parameters']:
            if param.get('in') == 'body':
                param_name = param.get('name', '')
                param_type = param.get('type', '').lower()
                
                if param_type == 'boolean':
                    body_params[param_name] = True
                elif param_type == 'integer' or param_type == 'number':
                    body_params[param_name] = 123
                else:
                    body_params[param_name] = "example_value"
        
        if body_params:
            curl += f" \\\n  -d '{json.dumps(body_params)}'"
    
    return curl

def create_python_example(endpoint):
    """Create a Python example for an endpoint."""
    path = endpoint['path']
    method = endpoint['method'].lower()
    
    # Replace path parameters with values
    for param in endpoint['parameters']:
        if param.get('in') == 'path':
            param_name = param.get('name', '')
            path = path.replace(f"{{{param_name}}}", "123")
    
    # Use the appropriate setup for the API
    if "github" in endpoint.get('source', '').lower() or "github" in path.lower():
        python = "import requests\n\n"
        python += "# GitHub API configuration\n"
        python += "base_url = 'https://api.github.com'\n"
        python += "github_token = 'YOUR_GITHUB_TOKEN'\n\n"
        python += "# Request headers\n"
        python += "headers = {\n"
        python += "    'Authorization': f'token {github_token}',\n"
        python += "    'Accept': 'application/vnd.github.v3+json'\n"
        python += "}\n\n"
    elif "kaggle" in endpoint.get('source', '').lower() or "kaggle" in path.lower():
        python = "from kaggle.api.kaggle_api_extended import KaggleApi\n\n"
        python += "# Initialize the Kaggle API\n"
        python += "api = KaggleApi()\n"
        python += "api.authenticate()\n\n"
        
        # Extract method name from path
        parts = path.split('/')
        if len(parts) >= 3:
            class_name = parts[2]
            method_name = parts[3] if len(parts) > 3 else ""
            
            python += f"# Call the {method_name} method\n"
            
            # Create method call with parameters
            params = []
            for param in endpoint['parameters']:
                param_name = param.get('name', '')
                param_type = param.get('type', '').lower()
                
                if param_type == 'boolean':
                    params.append(f"{param_name}=True")
                elif param_type == 'integer' or param_type == 'number':
                    params.append(f"{param_name}=123")
                else:
                    params.append(f"{param_name}='example'")
            
            python += f"result = api.{method_name}({', '.join(params)})\n\n"
            python += "# Process the result\n"
            python += "print(result)\n"
            
            return python
    else:
        python = "import requests\n\n"
        python += "# API configuration\n"
        python += "base_url = 'https://api.example.com'\n"
        python += "api_key = 'YOUR_API_KEY'\n\n"
        python += "# Request headers\n"
        python += "headers = {\n"
        python += "    'Authorization': f'Bearer {api_key}',\n"
        python += "    'Content-Type': 'application/json'\n"
        python += "}\n\n"
    
    # Add query parameters
    query_params = {}
    for param in endpoint['parameters']:
        if param.get('in') == 'query':
            param_name = param.get('name', '')
            param_type = param.get('type', '').lower()
            
            if param_type == 'boolean':
                query_params[param_name] = True
            elif param_type == 'integer' or param_type == 'number':
                query_params[param_name] = 123
            else:
                query_params[param_name] = "example"
    
    if query_params:
        python += "# Query parameters\n"
        python += f"params = {str(query_params)}\n\n"
    
    # Add request body
    body_params = {}
    if method in ['post', 'put', 'patch']:
        for param in endpoint['parameters']:
            if param.get('in') == 'body':
                param_name = param.get('name', '')
                param_type = param.get('type', '').lower()
                
                if param_type == 'boolean':
                    body_params[param_name] = True
                elif param_type == 'integer' or param_type == 'number':
                    body_params[param_name] = 123
                else:
                    body_params[param_name] = "example_value"
        
        if body_params:
            python += "# Request body\n"
            python += f"data = {str(body_params)}\n\n"
    
    # Make the request
    python += "# Send request\n"
    if method in ['get', 'delete'] and query_params:
        python += f"response = requests.{method}(f'{{base_url}}{path}', headers=headers, params=params)\n\n"
    elif method in ['get', 'delete']:
        python += f"response = requests.{method}(f'{{base_url}}{path}', headers=headers)\n\n"
    elif body_params:
        python += f"response = requests.{method}(f'{{base_url}}{path}', headers=headers, json=data)\n\n"
    else:
        python += f"response = requests.{method}(f'{{base_url}}{path}', headers=headers)\n\n"
    
    # Process the response
    python += "# Process response\n"
    python += "if response.status_code == 200:\n"
    python += "    data = response.json()\n"
    python += "    print(data)\n"
    python += "else:\n"
    python += "    print(f'Error: {response.status_code}')\n"
    python += "    print(response.text)\n"
    
    return python

def create_documentation_examples(endpoints):
    """Create documentation examples for fine-tuning from endpoints."""
    all_examples = []
    
    for endpoint in tqdm(endpoints, desc="Creating examples"):
        # Example 1: Complete documentation
        all_examples.append({
            "instruction": f"Write comprehensive API documentation for the following endpoint: {endpoint['method']} {endpoint['path']}",
            "response": format_full_documentation(endpoint),
            "meta": {
                "task": "complete_documentation",
                "endpoint": f"{endpoint['method']} {endpoint['path']}",
                "source": endpoint.get('source', '')
            }
        })
        
        # Example 2: Parameter documentation
        if endpoint['parameters']:
            all_examples.append({
                "instruction": f"Document all parameters for the endpoint: {endpoint['method']} {endpoint['path']}",
                "response": format_parameters_documentation(endpoint),
                "meta": {
                    "task": "parameter_documentation",
                    "endpoint": f"{endpoint['method']} {endpoint['path']}",
                    "source": endpoint.get('source', '')
                }
            })
        
        # Example 3: Response documentation
        if endpoint['responses']:
            all_examples.append({
                "instruction": f"Document the response format for the endpoint: {endpoint['method']} {endpoint['path']}",
                "response": format_response_documentation(endpoint),
                "meta": {
                    "task": "response_documentation",
                    "endpoint": f"{endpoint['method']} {endpoint['path']}",
                    "source": endpoint.get('source', '')
                }
            })
        
        # Example 4: Generate improvement suggestions
        # This creates examples where the model learns to suggest improvements to documentation
        all_examples.append({
            "instruction": f'''Review and suggest improvements for this API documentation:"response": f"""Here are suggestions to improve this API documentation:

1. Add clear examples showing how to use this endpoint in different programming languages
2. Provide more detailed explanations of parameter constraints and valid values 
3. Include information about rate limiting and authentication requirements
4. Document possible error responses and troubleshooting tips
5. Add links to related endpoints or resources''',
            "meta": {
                "task": "documentation_improvement",
                "endpoint": f"{endpoint['method']} {endpoint['path']}",
                "source": endpoint.get('source', '')
            }
        })
    
    return all_examples

def main():
    """Main function to build the API documentation dataset."""
    print("Building API Documentation Dataset for Fine-tuning")
    
    all_endpoints = []
    
    # 1. Download GitHub REST API specification
    github_spec = download_github_rest_api_spec()
    if github_spec:
        github_endpoints = process_openapi_spec(github_spec)
        all_endpoints.extend(github_endpoints)
    
    # 2. Process Kaggle API documentation
    kaggle_repo = clone_kaggle_api_repo()
    if kaggle_repo:
        kaggle_endpoints = extract_kaggle_api_docs(kaggle_repo)
        all_endpoints.extend(kaggle_endpoints)
    
    # 3. Add some popular OpenAPI specifications
    # Download Swagger Petstore specification
    petstore_url = "https://petstore.swagger.io/v2/swagger.json"
    print(f"Downloading Swagger Petstore specification from {petstore_url}")
    response = requests.get(petstore_url)
    if response.status_code == 200:
        petstore_path = "petstore_api.json"
        with open(petstore_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        petstore_endpoints = process_openapi_spec(petstore_path)
        all_endpoints.extend(petstore_endpoints)
    
    # 4. Create documentation examples
    print(f"Creating documentation examples from {len(all_endpoints)} endpoints")
    all_examples = create_documentation_examples(all_endpoints)
    
    # 5. Split into train/validation/test sets
    random.shuffle(all_examples)
    
    train_size = int(len(all_examples) * 0.8)
    val_size = int(len(all_examples) * 0.1)
    
    train_examples = all_examples[:train_size]
    val_examples = all_examples[train_size:train_size + val_size]
    test_examples = all_examples[train_size + val_size:]
    
    print(f"Dataset split - Train: {len(train_examples)}, Validation: {len(val_examples)}, Test: {len(test_examples)}")
    
    # 6. Create and save datasets
    output_dir = "api_docs_dataset"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "json"), exist_ok=True)
    
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)
    test_dataset = Dataset.from_list(test_examples)
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    print(f"Saving dataset to {output_dir}")
    dataset_dict.save_to_disk(output_dir)
    
    # Save sample JSON files for inspection
    with open(os.path.join(output_dir, "json", "train_sample.json"), 'w', encoding='utf-8') as f:
        json.dump(train_examples[:5], f, indent=2)
    
    with open(os.path.join(output_dir, "json", "validation_sample.json"), 'w', encoding='utf-8') as f:
        json.dump(val_examples[:5], f, indent=2)
    
    with open(os.path.join(output_dir, "json", "test_sample.json"), 'w', encoding='utf-8') as f:
        json.dump(test_examples[:5], f, indent=2)
    
    print("Dataset creation completed successfully!")
    print(f"Total examples: {len(all_examples)}")
    print(f"Sources: GitHub API, Kaggle API, Swagger Petstore")
    print(f"Dataset saved to: {output_dir}")

if __name__ == "__main__":
    main()