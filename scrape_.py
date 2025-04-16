import os
import json
import yaml
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import random
import datetime

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()
        return super().default(obj)
    

def find_openapi_files(directory):
    """Find all OpenAPI specification files (JSON and YAML) in a directory."""
    openapi_files = []
    
    for path in Path(directory).rglob("*"):
        if path.is_file() and path.suffix.lower() in ['.json', '.yaml', '.yml']:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Try to parse as JSON
                    if path.suffix.lower() == '.json':
                        data = json.loads(content)
                    else:
                        data = yaml.safe_load(content)
                    
                    # Check if this is an OpenAPI spec
                    if 'swagger' in data or 'openapi' in data:
                        openapi_files.append(str(path))
            except Exception as e:
                # Skip files that can't be parsed
                continue
    
    return openapi_files

def process_openapi_spec(file_path):
    """Extract API endpoints from an OpenAPI specification."""
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
                try:
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
                                    # Use the custom JSON encoder here
                                    try:
                                        if isinstance(media_info['example'], (dict, list)):
                                            example_json = json.dumps(
                                                media_info['example'], 
                                                indent=2,
                                                cls=DateTimeEncoder  # Use custom encoder
                                            )
                                        else:
                                            example_json = str(media_info['example'])
                                        
                                        response_info['content'][media_type] = {
                                            'example': example_json
                                        }
                                    except Exception as e:
                                        # In case of any serialization error, use a placeholder
                                        response_info['content'][media_type] = {
                                            'example': f"Example could not be displayed: {str(e)}"
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
                except Exception as e:
                    print(f"Error processing endpoint {path}/{method} in {file_path}: {str(e)}")
                    continue
    
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

def create_documentation_example(endpoint, task="complete"):
    """Create a documentation example for fine-tuning."""
    
    if task == "complete":
        # Complete documentation
        instruction = f"Write comprehensive API documentation for the following endpoint: {endpoint['method']} {endpoint['path']}"
        response = format_full_documentation(endpoint)
    elif task == "parameters":
        # Parameter documentation
        instruction = f"Document all parameters for the endpoint: {endpoint['method']} {endpoint['path']}"
        response = format_parameters_documentation(endpoint)
    elif task == "responses":
        # Response documentation
        instruction = f"Document the response format for the endpoint: {endpoint['method']} {endpoint['path']}"
        response = format_response_documentation(endpoint)
    elif task == "examples":
        # Usage examples
        instruction = f"Provide code examples for using the endpoint: {endpoint['method']} {endpoint['path']}"
        response = format_usage_example(endpoint)
    else:
        return None
    
    return {
        "instruction": instruction,
        "response": response,
        "meta": {
            "task": task,
            "endpoint": f"{endpoint['method']} {endpoint['path']}",
            "source": endpoint.get('source', '')
        }
    }



# Documentation formatting functions omitted for brevity - use the same ones from the previous script
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
    
    # Example usage section
    doc += "## Example Usage\n\n"
    
    # Create a curl example
    curl_example = create_curl_example(endpoint)
    doc += f"### cURL\n\n```bash\n{curl_example}\n```\n\n"
    
    # Create a Python example
    python_example = create_python_example(endpoint)
    doc += f"### Python\n\n```python\n{python_example}\n```\n\n"
    
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

def format_usage_example(endpoint):
    """Format usage examples for an endpoint."""
    doc = "## Usage Examples\n\n"
    
    # cURL example
    curl_example = create_curl_example(endpoint)
    doc += f"### cURL\n\n```bash\n{curl_example}\n```\n\n"
    
    # Python example
    python_example = create_python_example(endpoint)
    doc += f"### Python\n\n```python\n{python_example}\n```\n\n"
    
    # JavaScript example
    javascript_example = create_javascript_example(endpoint)
    doc += f"### JavaScript\n\n```javascript\n{javascript_example}\n```\n\n"
    
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
    
    base_url = "https://api.example.com"
    curl = f"curl -X {method} {base_url}{path}"
    
    # Add headers
    curl += ' \\\n  -H "Content-Type: application/json" \\\n  -H "Authorization: Bearer YOUR_API_KEY"'
    
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
    
    example = "import requests\n\n"
    example += "# API configuration\n"
    example += "base_url = 'https://api.example.com'\n"
    example += "api_key = 'YOUR_API_KEY'\n\n"
    
    example += "# Request headers\n"
    example += "headers = {\n"
    example += "    'Authorization': f'Bearer {api_key}',\n"
    example += "    'Content-Type': 'application/json'\n"
    example += "}\n\n"
    
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
        example += "# Query parameters\n"
        example += f"params = {str(query_params)}\n\n"
    
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
            example += "# Request body\n"
            example += f"data = {str(body_params)}\n\n"
    
    # Make the request
    example += "# Send request\n"
    if method in ['get', 'delete'] and query_params:
        example += f"response = requests.{method}(f'{{base_url}}{path}', headers=headers, params=params)\n\n"
    elif method in ['get', 'delete']:
        example += f"response = requests.{method}(f'{{base_url}}{path}', headers=headers)\n\n"
    elif body_params:
        example += f"response = requests.{method}(f'{{base_url}}{path}', headers=headers, json=data)\n\n"
    else:
        example += f"response = requests.{method}(f'{{base_url}}{path}', headers=headers)\n\n"
    
    # Process the response
    example += "# Process response\n"
    example += "if response.status_code == 200:\n"
    example += "    data = response.json()\n"
    example += "    print(data)\n"
    example += "else:\n"
    example += "    print(f'Error: {response.status_code}')\n"
    example += "    print(response.text)\n"
    
    return example

def create_javascript_example(endpoint):
    """Create a JavaScript example for an endpoint."""
    path = endpoint['path']
    method = endpoint['method'].lower()
    
    # Replace path parameters with values
    for param in endpoint['parameters']:
        if param.get('in') == 'path':
            param_name = param.get('name', '')
            path = path.replace(f"{{{param_name}}}", "123")
    
    example = "// API configuration\n"
    example += "const baseUrl = 'https://api.example.com';\n"
    example += "const apiKey = 'YOUR_API_KEY';\n\n"
    
    # Add query parameters
    query_params = {}
    for param in endpoint['parameters']:
        if param.get('in') == 'query':
            param_name = param.get('name', '')
            param_type = param.get('type', '').lower()
            
            if param_type == 'boolean':
                query_params[param_name] = true
            elif param_type == 'integer' or param_type == 'number':
                query_params[param_name] = 123
            else:
                query_params[param_name] = "example"
    
    if query_params and (method == 'get' or method == 'delete'):
        example += "// Query parameters\n"
        example += "const queryParams = new URLSearchParams({\n"
        for name, value in query_params.items():
            if isinstance(value, str):
                example += f"  {name}: '{value}',\n"
            else:
                example += f"  {name}: {value},\n"
        example += "});\n\n"
    
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
    
    # Function to make the request
    example += "// Function to make the API request\n"
    example += "async function makeApiRequest() {\n"
    example += "  try {\n"
    
    # Request options
    example += "    const options = {\n"
    example += f"      method: '{method.toUpperCase()}',\n"
    example += "      headers: {\n"
    example += "        'Authorization': `Bearer ${apiKey}`,\n"
    example += "        'Content-Type': 'application/json'\n"
    example += "      }\n"
    
    if body_params:
        example += "      ,body: JSON.stringify(" + json.dumps(body_params) + ")\n"
    
    example += "    };\n\n"
    
    # Build URL
    if query_params and (method == 'get' or method == 'delete'):
        example += f"    const url = `${{baseUrl}}{path}?${{queryParams}}`;\n"
    else:
        example += f"    const url = `${{baseUrl}}{path}`;\n"
    
    # Make request
    example += "    const response = await fetch(url, options);\n"
    example += "    const data = await response.json();\n\n"
    
    # Handle response
    example += "    if (response.ok) {\n"
    example += "      console.log('Success:', data);\n"
    example += "      return data;\n"
    example += "    } else {\n"
    example += "      console.error('Error:', response.status, data);\n"
    example += "      throw new Error(`API request failed with status ${response.status}`);\n"
    example += "    }\n"
    
    example += "  } catch (error) {\n"
    example += "    console.error('Request failed:', error);\n"
    example += "    throw error;\n"
    example += "  }\n"
    example += "}\n\n"
    
    # Call the function
    example += "// Call the API\n"
    example += "makeApiRequest()\n"
    example += "  .then(data => console.log('Processing data:', data))\n"
    example += "  .catch(error => console.error('Error occurred:', error));\n"
    
    return example

def main():
    parser = argparse.ArgumentParser(description="Process OpenAPI specifications into a dataset")
    parser.add_argument("--input_dir", default="openapi_specs", help="Directory containing OpenAPI specifications")
    parser.add_argument("--output_dir", default="./api_docs_dataset", help="Output directory for the dataset")
    
    args = parser.parse_args()
    
    # Find OpenAPI files
    print(f"Finding OpenAPI specifications in {args.input_dir}")
    openapi_files = find_openapi_files(args.input_dir)
    print(f"Found {len(openapi_files)} OpenAPI specification files")
    
    if not openapi_files:
        print("No OpenAPI files found. Make sure you've downloaded some specifications first.")
        return
    
    # Process specifications
    all_endpoints = []
    for file_path in tqdm(openapi_files, desc="Processing OpenAPI files"):
        try:
            endpoints = process_openapi_spec(file_path)
            all_endpoints.extend(endpoints)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    print(f"Extracted {len(all_endpoints)} API endpoints")
    
    # Create documentation examples
    all_examples = []
    
    for endpoint in tqdm(all_endpoints, desc="Creating examples"):
        # Complete documentation
        example = create_documentation_example(endpoint, "complete")
        if example:
            all_examples.append(example)
        
        # Parameter documentation
        if endpoint['parameters']:
            example = create_documentation_example(endpoint, "parameters")
            if example:
                all_examples.append(example)
        
        # Response documentation
        if endpoint['responses']:
            example = create_documentation_example(endpoint, "responses")
            if example:
                all_examples.append(example)
        
        # Usage examples
        example = create_documentation_example(endpoint, "examples")
        if example:
            all_examples.append(example)
    
    print(f"Created {len(all_examples)} documentation examples")
    
    # Split dataset
    random.shuffle(all_examples)
    
    train_size = int(len(all_examples) * 0.8)
    val_size = int(len(all_examples) * 0.1)
    
    train_examples = all_examples[:train_size]
    val_examples = all_examples[train_size:train_size + val_size]
    test_examples = all_examples[train_size + val_size:]
    
    print(f"Train: {len(train_examples)}, Validation: {len(val_examples)}, Test: {len(test_examples)}")
    
    # Create and save datasets
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "json"), exist_ok=True)
    
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)
    test_dataset = Dataset.from_list(test_examples)
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    print(f"Saving dataset to {args.output_dir}")
    dataset_dict.save_to_disk(args.output_dir)
    
    # Save JSON for inspection

    with open(os.path.join(args.output_dir, "json", "train.json"), 'w', encoding='utf-8') as f:
        json.dump(train_examples[:100], f, indent=2, cls=DateTimeEncoder)  # Use the custom encoder
    
    print("Dataset creation completed successfully!")

if __name__ == "__main__":
    main()