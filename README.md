# **LM Studio Chat Interface with Vector Search**

A powerful chat interface for LM Studio that enables semantic search and context-aware interactions with your documents. This application combines the capabilities of LM Studio with advanced embedding features for more intelligent document analysis and retrieval.

## **✨ Features**

- **Semantic Search**: Uses embeddings to find relevant context across your documents
- **Easy File Upload**: Upload .txt files through a simple interface
- **Intelligent File Referencing**: Use @mentions with semantic understanding
- **Vector-Based Document Storage**: Efficient document storage and retrieval using ChromaDB
- **Real-time Status Updates**: Clear feedback on system initialization and readiness
- **Dynamic Model Selection**: Automatically uses available models from your LM Studio instance
- **Configurable Connection**: Easy IP address and port configuration for LM Studio
- **Responsive Design**: Works well on desktop and mobile devices

## **🚀 Quick Start**

### **Prerequisites**

- Python 3.x
- LM Studio installed and running
- Modern web browser
- 2GB+ free disk space (for embedding model)

### **Installation**

1. **Clone the repository:**
   ```bash
   git clone [repository-url]
   cd chat-with-embedding
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download the embedding model:**
   ```bash
   mkdir -p embedding/mxbai-embed-large-v1
   # Download model files to embedding/mxbai-embed-large-v1/
   ```

### **Running the Application**

1. **Start LM Studio** and load your preferred model

2. **Run the Flask application:**
   ```bash
   python app.py
   ```

3. **Access the interface** at http://localhost:5000

4. **Wait for initialization** - The system will indicate when it's ready

## **📝 Usage Guide**

### **Document Management**

1. **Uploading Documents:**
   - Wait for the system to indicate "Ready"
   - Click the file input in "Upload Text Documents"
   - Select a .txt file
   - Click "Upload"
   - The system will process and embed the document automatically

2. **Referencing Documents:**
   - **Direct @mention:** `What's in @document.txt?`
   - **Copy Reference:** Click "Copy Reference" next to any file
   - The system will find relevant context even from other documents

### **Chat Interactions**

1. **Simple Queries:**
   - Just type your question and send
   - The AI will respond based on the conversation context

2. **Document-Aware Queries:**
   - Reference documents using @mentions
   - The system will:
     - Find relevant sections across all documents
     - Include context in the prompt
     - Generate informed responses

### **Configuration**

1. **LM Studio Connection:**
   - IP Address (default: localhost)
   - Port (default: 1234)
   - Click "Save Configuration"

2. **System Status:**
   - Watch the status bar for:
     - Initialization progress
     - System readiness
     - Any error states

## **🔍 Example Queries**

- ```What's the main topic in @document.txt?```
- ```Find similar content to @report.txt```
- ```Summarize the key points from @file1.txt```
- ```What do my documents say about [topic]?```
- ```Compare the perspectives in @file1.txt and @file2.txt```

## **🛠️ Technical Details**

- **Frontend:**
  - Modern HTML5/CSS3/JavaScript
  - Real-time status updates
  - Responsive design

- **Backend:**
  - Flask server
  - Asynchronous initialization
  - ChromaDB for vector storage
  - MiniLM embedding model

- **Vector Search:**
  - Document chunking and embedding
  - Semantic similarity search
  - Context-aware prompt construction

## **🔧 Troubleshooting**

1. **Initialization Issues:**
   - Verify embedding model is downloaded correctly
   - Check disk space for vector storage
   - Monitor status bar for error messages

2. **Connection Issues:**
   - Verify LM Studio is running
   - Check connection settings
   - Ensure no firewall blocking

3. **Search Issues:**
   - Verify documents are uploaded successfully
   - Check file encodings (use UTF-8)
   - Monitor server logs for embedding errors

## **📁 Project Structure**
```
chat_with_embedding/
├── app.py                  # Main Flask application
├── requirements.txt        # Dependencies
├── embedding/             # Embedding-related modules
│   ├── __init__.py
│   ├── model.py           # Embedding model wrapper
│   ├── vector_store.py    # Vector storage interface
│   └── mxbai-embed-large-v1/  # Embedding model files
├── static/               # Static assets
│   └── style.css
├── templates/            # HTML templates
│   └── index.html
├── uploads/             # Document storage
└── vector_db/          # ChromaDB storage
```

## **🚀 Performance Tips**

1. **Document Processing:**
   - Keep individual files under 1MB for best performance
   - Use plain text format (.txt)
   - Ensure proper text encoding (UTF-8)

2. **Search Optimization:**
   - Be specific in your queries
   - Use descriptive @mentions
   - Let the system finish initialization before uploading

## **🤝 Contributing**

Contributions welcome:
- Bug reports
- Feature requests
- Documentation improvements
- Pull requests

## **📜 License**

This project is for educational purposes.

## **👏 Acknowledgments**

- LM Studio team
- Sentence Transformers library
- ChromaDB team
- All contributors

---

**Note:** This interface requires LM Studio for chat completions and uses the MiniLM model for embeddings.