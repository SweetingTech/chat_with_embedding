from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import requests
import json
import re
from typing import List, Dict
from embedding import VectorStore
import threading
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['VECTOR_DB'] = 'vector_db/'

# Global flags for initialization status
vector_store = None
is_initializing = False
initialization_complete = False
initialization_error = None

def initialize_vector_store():
    """Initialize the vector store in a separate thread."""
    global vector_store, is_initializing, initialization_complete, initialization_error
    try:
        print("\n=== Starting Vector Store Initialization ===")
        print("This may take a few moments while the embedding model loads...")
        vector_store = VectorStore(persist_directory=app.config['VECTOR_DB'])
        print("=== Vector Store Initialization Complete ===\n")
        initialization_complete = True
    except Exception as e:
        initialization_error = str(e)
        print(f"Error during initialization: {str(e)}")
    finally:
        is_initializing = False

# Start initialization in background thread
threading.Thread(target=initialize_vector_store).start()
is_initializing = True

# Create necessary directories if they don't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['VECTOR_DB']):
    os.makedirs(app.config['VECTOR_DB'])

def get_vector_store():
    """Get the vector store, waiting for initialization if necessary."""
    global vector_store, is_initializing, initialization_complete, initialization_error
    
    # Wait for initialization (with timeout)
    timeout = 60  # 60 seconds timeout
    start_time = time.time()
    while is_initializing and (time.time() - start_time) < timeout:
        time.sleep(0.1)
    
    if initialization_error:
        raise Exception(f"Vector store initialization failed: {initialization_error}")
    
    if not initialization_complete:
        raise Exception("Vector store initialization timed out")
        
    return vector_store

@app.route('/initialization_status')
def initialization_status():
    """Return the current initialization status."""
    return jsonify({
        'is_initializing': is_initializing,
        'is_complete': initialization_complete,
        'error': initialization_error
    })

def create_augmented_prompt(query: str, relevant_chunks: List[Dict]) -> str:
    """Create an augmented prompt with relevant context."""
    if not relevant_chunks:
        return query
        
    context_str = "\n\n".join([
        f"From {chunk['filename']} (similarity: {chunk['similarity']:.2f}):\n{chunk['content']}"
        for chunk in sorted(relevant_chunks, key=lambda x: x['similarity'], reverse=True)
    ])
    
    return f"""I found the following relevant information from the documents:

{context_str}

Based on this context, please help with the following query:
{query}"""

def extract_document_name(message):
    """Extract document name from @mentions in the message."""
    matches = re.findall(r'@([\w.-]+\.txt)', message)
    if matches:
        return matches[0]
    return None

@app.route('/')
def index():
    """Render the main page with a list of uploaded files."""
    uploaded_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                     if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))
                     and f.endswith('.txt')]
    return render_template('index.html', uploaded_files=uploaded_files)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads and add them to the vector store."""
    if 'file' not in request.files:
        return redirect(url_for('index'))
        
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
        
    if file.filename.endswith('.txt'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Add to vector store
        try:
            store = get_vector_store()
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                store.add_document(file.filename, content)
            print(f"Successfully processed file: {file.filename}")
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            
    return redirect(url_for('index'))

@app.route('/files', methods=['GET'])
def list_files():
    """Return a list of uploaded files."""
    files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
             if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))
             and f.endswith('.txt')]
    return jsonify({'files': files})

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages and integrate with LM Studio."""
    data = request.json
    user_input = data['message']
    ip_address = data.get('ip_address', 'localhost')
    port = data.get('port', '1234')
    
    print(f"\nReceived user input: {user_input}")
    
    document_name = extract_document_name(user_input)
    prompt = user_input  # Default to original input
    
    # Only use vector store if a document is referenced
    if document_name:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], document_name)
        print(f"Looking for file at: {filepath}")
        
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as file:
                document_content = file.read()
            print(f"File content read successfully")
            
            # Clean up the user input by removing the @mention
            clean_input = user_input.replace(f'@{document_name}', f'the file {document_name}')
            
            # Add the document content directly if it's small enough
            if len(document_content) < 2000:  # Arbitrary threshold
                prompt = f"""The content of {document_name} is:

{document_content}

{clean_input}"""
            else:
                # For larger documents, use vector search to find relevant parts
                try:
                    store = get_vector_store()
                    relevant_chunks = store.query_similar(clean_input)
                    prompt = create_augmented_prompt(clean_input, relevant_chunks)
                except Exception as e:
                    print(f"Error in vector search: {str(e)}")
                    # Fallback to direct content if vector search fails
                    prompt = f"Content of {document_name}: {document_content}\n\n{clean_input}"
        else:
            print(f"File not found at: {filepath}")
            return jsonify({'response': f"The document '{document_name}' was not found in the uploads folder."})
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. When analyzing file contents, focus on the relevant context provided and incorporate it into your responses."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    print("\nSending messages to LM Studio...")

    lm_studio_url = f"http://{ip_address}:{port}/v1/chat/completions"

    try:
        model_info_response = requests.get(f"http://{ip_address}:{port}/v1/models")
        if model_info_response.status_code != 200:
            return jsonify({'response': 'Error: Unable to get model information from LM Studio'})
        
        model_info = model_info_response.json()
        if not model_info.get('data'):
            return jsonify({'response': 'Error: No models available in LM Studio'})
        
        model_id = model_info['data'][0]['id']
        
        response = requests.post(
            lm_studio_url,
            headers={'Content-Type': 'application/json'},
            data=json.dumps({
                "model": model_id,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": -1,
                "stream": False
            }),
            timeout=30
        )

        if response.status_code == 200:
            response_data = response.json()
            response_text = response_data['choices'][0]['message']['content'].strip()
            if not response_text:
                response_text = "Received empty response from LM Studio"
        else:
            response_text = f"Error: {response.status_code} - {response.text}"

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to LM Studio: {str(e)}")
        response_text = f"Error connecting to LM Studio: {str(e)}"

    return jsonify({'response': response_text})

if __name__ == '__main__':
    print("\n=== Starting Server ===")
    print("Vector store initialization has begun in the background.")
    print("The embedding model is loading and will be ready shortly.")
    print("You can start using the chat interface immediately.")
    print("Document-related features will be available once initialization is complete.")
    print("======================\n")
    app.run(debug=True)