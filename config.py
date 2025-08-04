# config.py
import os
from dotenv import load_dotenv

# Load environment variables from a .env file for local development
load_dotenv()

# --- Core File & Directory Paths ---
# Assumes data is placed in a sibling 'data' directory.
# Example: project_root/data/your_data.csv
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CSV_FILE_NAME = "Llama-3.3-70B-Instruct-Turbo-Free (2).csv"
CSV_FILE_PATH = os.path.join(DATA_DIR, CSV_FILE_NAME)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
PROMPT_FILE_PATH = os.path.join(os.path.dirname(__file__), 'prompts', 'fides_score_judge.txt')

# --- Evaluation Parameters ---
# Set to a small integer for testing, or None to process all rows.
# WARNING: Processing all rows can be time-consuming and costly.
MAX_ROWS_TO_PROCESS: int | None = 5
CONFIDENCE_THRESHOLD: float = 6.0  # Scores < 6.0 are considered "non-compliant" (label 1)

# --- LLM-as-a-Judge Configuration ---
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
LLM_JUDGE_MODEL: str = "gpt-4o"
API_REQUEST_TIMEOUT_SECONDS: int = 60
API_CALL_DELAY_SECONDS: float = 1.0  # Delay between API calls to respect rate limits
