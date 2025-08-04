# XAFC Evaluation Suite

## 1. Overview

The **Explainable Ad-Compliance Framework (XAFC) Evaluation Suite** is a professional-grade Python application designed to rigorously assess the performance of language models in ad-tech compliance checking.

This suite performs two primary modes of analysis:

1.  **Classification Performance Analysis**: Calculates standard metrics (F1, Precision, Recall) and generates a visual confusion matrix for the model's binary compliance predictions.
2.  **Explanation Quality Analysis**: Implements a true **LLM-as-a-Judge** pattern using a powerful external model (e.g., GPT-4o) to evaluate the quality of generated explanations against the **FIDES-Score** rubric.

This project is built to be modular, configurable, and easily extendable.

---

## 2. Project Structure

The project uses a clean and maintainable structure.

---

## 3. Setup and Installation

**Prerequisites:**

* Python 3.8+
* Git
* An OpenAI API Key

Follow these steps to get the project running on your local machine.

### Step 1: Set Up the Project Environment

Run the following commands in your terminal. This single block will clone the repository, navigate into the directory, create a virtual environment, activate it, and install all necessary dependencies.

```
# Clone the repository (replace YOUR_USERNAME)
git clone [https://github.com/YOUR_USERNAME/xafc-evaluation-suite.git](https://github.com/YOUR_USERNAME/xafc-evaluation-suite.git)

# Navigate into the project directory
cd xafc-evaluation-suite

# Create a Python virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install all required libraries
pip install -r requirements.txt

```

### Step 2: Configure API Key and Settings
Create ```.env``` file: Create a new file named ```.env``` in the project root (```xafc_evaluation_suite/```) and add your API key to it:
```
# .env
OPENAI_API_KEY="sk-..."
```
Verify ```config.py```: Open ```config.py``` and ensure the ```CSV_FILE_NAME``` points to your data file located in the ```data/ directory```. Adjust ```MAX_ROWS_TO_PROCESS``` as needed.
---

## 4. How to Run
Execute the main script from the project's root directory:
```python main.py```

The script will run both the classification analysis and the FIDES-Score evaluation, printing results to the console and saving the confusion matrix chart to the results/ directory.
