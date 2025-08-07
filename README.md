Intelligent Code Evaluation Platform
Overview
This project is an advanced code evaluation platform that uses Retrieval-Augmented Generation (RAG) and Abstract Syntax Trees (AST) to provide deep insights into code submissions. It analyzes code to calculate complexity, retrieve similar problems from a vector database, and offer intelligent feedback beyond simple pass/fail checks.

Core Features
RAG-Powered Feedback: Provides context-aware suggestions by retrieving similar code examples.

AST-Based Complexity Analysis: Accurately calculates time and space complexity by parsing the code's structure comparing it with optimal solution.

Similar Problem Retrieval: Helps users practice by identifying problems with similar patterns and practice.

Vector Database Integration: Uses a vector database for efficient similarity search on code embeddings.



How It Works
When a user submits code, the platform parses it into an Abstract Syntax Tree (AST) to analyze its complexity. It then creates a vector embedding of the code and queries a database to find similar, pre-existing problems. This retrieved information, along with the complexity analysis, is used by a Large Language Model (LLM) to generate a comprehensive evaluation and suggest improvements.

Technology Stack
Piston API
Streamlit
Vector Database:FAISS
AST Parsing: Python's built-in ast module





Contributing
Contributions are welcome. Please fork the repository, create a feature branch, and open a pull request with your changes.

License
This project is licensed under the MIT License.
