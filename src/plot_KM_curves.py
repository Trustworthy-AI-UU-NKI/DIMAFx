import os
import argparse
import numpy as np
import pickle
import pandas as pd

from utils.general_utils import set_seed

from lifelines import CoxPHFitter, KaplanMeierFitter
import matplotlib.pyplot as plt

def update_risk_dict(current, new):
    """ Add new values to dictionary containing lists. """
    for item, value in new.items():
        if item not in current.keys(): 
            current[item] = value
        else:
            current[item] = np.concatenate((current[item], value), axis=0)
    
    return current

def plot_km_curves(all_data, model_name, result_dir):
    n_datasets = len(all_data)
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(6 * n_datasets, 5), sharex=True)
    # For every dataset
    for ax, (dataset_name, pooled_km_data) in zip(axes, all_data.items()):
        if pooled_km_data: 
            times = pooled_km_data["Time"]
            events = np.array([1 if x == 0 else 0 for x in pooled_km_data["Censorship"]]) # CHEK IF IT IS THE CORRECT ONE
            scores = pooled_km_data["Risk scores"]
            
            median_score = np.median(scores)
            high_risk = scores > median_score
            low_risk = scores < median_score   

            cph = CoxPHFitter()
            df = pd.DataFrame({
                "time": np.concatenate([times[high_risk], times[low_risk]]),
                "event": np.concatenate([events[high_risk], events[low_risk]]),
                "group": np.concatenate([np.ones(high_risk.sum()), np.zeros(low_risk.sum())]) 
            })
            cph.fit(df, duration_col="time", event_col="event")

            hr = cph.hazard_ratios_["group"]
            p_val = cph.summary.loc["group", "p"]

            # KM plot
            kmf = KaplanMeierFitter()
            for group, label_name, colour in zip([high_risk, low_risk], ["High Risk", "Low Risk"], ['red', 'blue']):
                kmf.fit(times[group], events[group], label=label_name)
                kmf.plot(ci_show=True, color=colour, show_censors=True, linewidth=2, ax=ax)
            ax.text(
                0.10, 0.10,
                f"HR = {hr:.3f}\np = {p_val:.2e}",
                transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='white'),
                fontsize=13
            )
            ax.set_ylim(0, 1.1)
            # Axis labels and title
            ax.set_ylabel("Disease-specific survival probability", fontsize=13)
            ax.set_xlabel("Time (days)", fontsize=13)
            ax.set_title(dataset_name.upper().replace("_", " "), fontsize=15, fontweight='bold')  
            ax.legend(loc='upper right', ncol=2, frameon=False,  fontsize=13)

    fig.text(
        -0.02, 0.5,
        model_name.replace("_", " "),
        va='center', rotation='vertical',
        fontsize=15, fontweight='bold'
    )
    plt.tight_layout()

    result_dir = os.path.join(result_dir, "KM_curves")
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig(os.path.join(result_dir, f"{model_name}.pdf"), format="pdf", bbox_inches="tight")
    plt.show()

def get_results_over_all_folds(result_dir):
  """ Get the predicted test risks. """
  risk_dict = dict()
  for i in range(5):
    fold_dir = os.path.join(result_dir, f"Fold_{i}/post_training/predicted_risk_scores_test.pkl")
    if os.path.exists(fold_dir):
        with open(fold_dir, 'rb') as f:
            risk_dict_fold = pickle.load(f)

        risk_dict = update_risk_dict(risk_dict, risk_dict_fold)
  
  return risk_dict
  

def main(args):
    set_seed(args.seed)
    all_datasets = ['dss_survival_brca', 'dss_survival_blca', 'dss_survival_luad', 'dss_survival_kirc']

    all_types_km_data = {}
    for data_type in all_datasets:
        result_dir = os.path.join(args.result_dir, data_type, args.exp_code)
        # Get risks
        pooled_km_data = get_results_over_all_folds(result_dir)
        new_data_type = ('tcga ' + data_type.split('_')[-1]).upper()
        all_types_km_data[new_data_type] = pooled_km_data

    plot_km_curves(all_types_km_data, args.exp_code, args.result_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot KM curves')
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--result_dir', default='results',help='results directory')
    parser.add_argument('--exp_code', type=str, default='DIMAFx', help='experiment code for saving results')

    args = parser.parse_args()
    main(args)