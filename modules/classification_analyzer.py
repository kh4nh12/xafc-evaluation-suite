# modules/classification_analyzer.py
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, Any

def _get_binary_label(data: Dict[str, Any], threshold: float) -> int:
    """Extracts confidence score and derives a binary label."""
    try:
        confidence_str = data.get("overall_compliance", {}).get("overall_confidence", "0")
        score_part = str(confidence_str).split('/')[0]
        score = float(score_part)
        return 1 if score < threshold else 0
    except (ValueError, TypeError, AttributeError):
        logging.warning(f"Could not parse confidence score from data: {data}. Defaulting to label 0.")
        return 0

def analyze_classification_performance(df: pd.DataFrame, config):
    """Calculates, prints, and visualizes classification performance metrics."""
    logging.info("Starting classification performance analysis.")
    
    df['true_label'] = df['compliance_data'].apply(lambda x: _get_binary_label(x, config.CONFIDENCE_THRESHOLD))
    df['pred_label'] = df['llm_output_data'].apply(lambda x: _get_binary_label(x, config.CONFIDENCE_THRESHOLD))

    true_labels = df['true_label']
    pred_labels = df['pred_label']
    
    print("\n" + "="*50)
    print("📊 CLASSIFICATION PERFORMANCE ANALYSIS")
    print("="*50)
    
    report = classification_report(true_labels, pred_labels, zero_division=0, target_names=["Compliant (0)", "Non-Compliant (1)"])
    print(report)

    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Predicted Compliant", "Predicted Non-Compliant"],
                yticklabels=["Actual Compliant", "Actual Non-Compliant"])
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(config.RESULTS_DIR, 'confusion_matrix.png')
    
    try:
        plt.savefig(save_path)
        logging.info(f"Confusion matrix plot saved successfully to: {save_path}")
    except Exception as e:
        logging.error(f"Failed to save confusion matrix plot: {e}")
