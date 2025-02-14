# Smart Adapt API

## Optimizing Domain-Specific Knowledge Integration Through Automated Techniques in Large Language Models

Smart Adapt is a parametric optimized approach designed to detect key dataset characteristics for effective LLM training on downstream tasks. It includes LoRA and HNSW hyperparameter detection using simulation for LLM training and real-time information retrieval.

## Minimum System Requirements
- **RAM:** 16GB or higher
- **Storage:** 30GB or Higher (SSD preferred)
- **Docker:** Configured
- **Jupyter Kernel:** Installed
- **Internet Connection:** Required to download the LLM
- **Python:** Version 3.11 installed

## Instructions
1. Create a Python virtual environment:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows use 'venv\\Scripts\\activate'
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook runner.ipynb
   ```
