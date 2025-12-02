"""
Visualization functions for inference results
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List


def visualize_inference_results(
    results_json_path: str,
    output_path: Optional[str] = None,
    figsize: tuple = (15, 10),
    dpi: int = 100
):
    """
    Visualize inference results: plot MSE and fraud probability with color coding based on ground truth
    
    Args:
        results_json_path: Path to the JSON file containing inference results
        output_path: Path to save the figure. If None, will be saved in the same directory as JSON file
        figsize: Figure size (width, height)
        dpi: Resolution of the figure
    
    Returns:
        fig: matplotlib figure object
    """
    # Load results
    with open(results_json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Extract data
    results_list = results['results']
    csv_file = results.get('csv_file', 'Unknown')
    avg_mse = results.get('average_sequence_mse', 0.0)
    
    # Extract time steps, MSE, fraud probabilities, and ground truth labels
    time_steps = []
    mse_values = []
    fraud_probs = []
    ground_truth_labels = []
    has_ground_truth = False
    
    for result in results_list:
        time_steps.append(result['predicted_step'])
        mse_values.append(result['sequence_mse'])
        fraud_probs.append(result['judge_model']['fraud_probability'])
        
        # Check if ground truth fraud label exists
        if 'ground_truth_fraud' in result:
            has_ground_truth = True
            ground_truth_labels.append(result['ground_truth_fraud'])
        else:
            ground_truth_labels.append(None)
    
    # Convert to numpy arrays
    time_steps = np.array(time_steps)
    mse_values = np.array(mse_values)
    fraud_probs = np.array(fraud_probs)
    ground_truth_labels = np.array(ground_truth_labels)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi)
    
    # Plot 1: MSE over time
    if has_ground_truth:
        # Color code based on ground truth fraud label
        colors = ['red' if label == 1 else 'blue' for label in ground_truth_labels]
        ax1.scatter(time_steps, mse_values, c=colors, alpha=0.6, s=20, label='MSE')
        ax1.plot(time_steps, mse_values, alpha=0.3, linewidth=1, color='gray')
    else:
        # No ground truth, use default color
        ax1.plot(time_steps, mse_values, linewidth=1.5, color='blue', label='MSE')
        ax1.scatter(time_steps, mse_values, alpha=0.6, s=20, color='blue')
    
    ax1.set_xlabel('Time Step (Predicted Step)', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.set_title(f'Sequence Model MSE Over Time\n(CSV: {Path(csv_file).name}, Avg MSE: {avg_mse:.4f})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add color legend if ground truth is available
    if has_ground_truth:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Ground Truth: Fraud'),
            Patch(facecolor='blue', label='Ground Truth: Normal')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
    
    # Plot 2: Fraud Probability over time
    if has_ground_truth:
        # Color code based on ground truth fraud label
        colors = ['red' if label == 1 else 'blue' for label in ground_truth_labels]
        ax2.scatter(time_steps, fraud_probs, c=colors, alpha=0.6, s=20, label='Fraud Probability')
        ax2.plot(time_steps, fraud_probs, alpha=0.3, linewidth=1, color='gray')
    else:
        # No ground truth, use default color
        ax2.plot(time_steps, fraud_probs, linewidth=1.5, color='blue', label='Fraud Probability')
        ax2.scatter(time_steps, fraud_probs, alpha=0.6, s=20, color='blue')
    
    # Add threshold line at 0.5
    ax2.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Threshold (0.5)')
    
    ax2.set_xlabel('Time Step (Predicted Step)', fontsize=12)
    ax2.set_ylabel('Fraud Probability', fontsize=12)
    ax2.set_title('Judge Model Fraud Probability Over Time', fontsize=14)
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add color legend if ground truth is available
    if has_ground_truth:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Ground Truth: Fraud'),
            Patch(facecolor='blue', label='Ground Truth: Normal'),
            plt.Line2D([0], [0], color='orange', linestyle='--', label='Threshold (0.5)')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save figure
    if output_path is None:
        # Save in the same directory as JSON file with .png extension
        json_path = Path(results_json_path)
        output_path = json_path.parent / f"{json_path.stem}_visualization.png"
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    return fig


def visualize_inference_results_from_dict(
    results_dict: Dict,
    output_path: Optional[str] = None,
    figsize: tuple = (15, 10),
    dpi: int = 100
):
    """
    Visualize inference results from a dictionary (same as from JSON file)
    
    Args:
        results_dict: Dictionary containing inference results
        output_path: Path to save the figure. If None, will be saved as 'inference_visualization.png'
        figsize: Figure size (width, height)
        dpi: Resolution of the figure
    
    Returns:
        fig: matplotlib figure object
    """
    # Extract data
    results_list = results_dict['results']
    csv_file = results_dict.get('csv_file', 'Unknown')
    avg_mse = results_dict.get('average_sequence_mse', 0.0)
    
    # Extract time steps, MSE, fraud probabilities, and ground truth labels
    time_steps = []
    mse_values = []
    fraud_probs = []
    ground_truth_labels = []
    has_ground_truth = False
    
    for result in results_list:
        time_steps.append(result['predicted_step'])
        mse_values.append(result['sequence_mse'])
        fraud_probs.append(result['judge_model']['fraud_probability'])
        
        # Check if ground truth fraud label exists
        if 'ground_truth_fraud' in result:
            has_ground_truth = True
            ground_truth_labels.append(result['ground_truth_fraud'])
        else:
            ground_truth_labels.append(None)
    
    # Convert to numpy arrays
    time_steps = np.array(time_steps)
    mse_values = np.array(mse_values)
    fraud_probs = np.array(fraud_probs)
    ground_truth_labels = np.array(ground_truth_labels)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi)
    
    # Plot 1: MSE over time
    if has_ground_truth:
        # Color code based on ground truth fraud label
        colors = ['red' if label == 1 else 'blue' for label in ground_truth_labels]
        ax1.scatter(time_steps, mse_values, c=colors, alpha=0.6, s=20, label='MSE')
        ax1.plot(time_steps, mse_values, alpha=0.3, linewidth=1, color='gray')
    else:
        # No ground truth, use default color
        ax1.plot(time_steps, mse_values, linewidth=1.5, color='blue', label='MSE')
        ax1.scatter(time_steps, mse_values, alpha=0.6, s=20, color='blue')
    
    ax1.set_xlabel('Time Step (Predicted Step)', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.set_title(f'Sequence Model MSE Over Time\n(CSV: {Path(csv_file).name}, Avg MSE: {avg_mse:.4f})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add color legend if ground truth is available
    if has_ground_truth:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Ground Truth: Fraud'),
            Patch(facecolor='blue', label='Ground Truth: Normal')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
    
    # Plot 2: Fraud Probability over time
    if has_ground_truth:
        # Color code based on ground truth fraud label
        colors = ['red' if label == 1 else 'blue' for label in ground_truth_labels]
        ax2.scatter(time_steps, fraud_probs, c=colors, alpha=0.6, s=20, label='Fraud Probability')
        ax2.plot(time_steps, fraud_probs, alpha=0.3, linewidth=1, color='gray')
    else:
        # No ground truth, use default color
        ax2.plot(time_steps, fraud_probs, linewidth=1.5, color='blue', label='Fraud Probability')
        ax2.scatter(time_steps, fraud_probs, alpha=0.6, s=20, color='blue')
    
    # Add threshold line at 0.5
    ax2.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Threshold (0.5)')
    
    ax2.set_xlabel('Time Step (Predicted Step)', fontsize=12)
    ax2.set_ylabel('Fraud Probability', fontsize=12)
    ax2.set_title('Judge Model Fraud Probability Over Time', fontsize=14)
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add color legend if ground truth is available
    if has_ground_truth:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Ground Truth: Fraud'),
            Patch(facecolor='blue', label='Ground Truth: Normal'),
            plt.Line2D([0], [0], color='orange', linestyle='--', label='Threshold (0.5)')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save figure
    if output_path is None:
        output_path = 'inference_visualization.png'
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    return fig




