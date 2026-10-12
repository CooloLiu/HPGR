"""HPGR entry point (Pattern Recognition version, tabular one-class classification only).

Pipeline:
    load .mat (last column = label) -> fit StandardScaler on normal samples only -> transform all
    -> X_train = normal samples, X_eval = all samples
    -> hierarchical probabilistic granular-ball splitting
    -> entropy-weighted accumulation of the per-level scores (use_ball_entropy_weighting=True),
       or final-level scores only (False)
    -> AUC / AP / F1 + confusion matrix + FPR / FNR
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from hpgr.config import load_config
from hpgr.data import load_dataset
from hpgr.metrics import evaluate_anomaly_detection
from hpgr.model import ProbabilisticGranularBallSystem


def main(args, X_train, X_eval, labels):
    labels_eval = labels

    n_anomalies = int(np.sum(labels != 0))
    total_samples = len(labels)
    is_anomaly_detection_task = n_anomalies > 0

    if is_anomaly_detection_task:
        print(f"Task: anomaly detection ({n_anomalies} / {total_samples} anomalies, "
              f"{n_anomalies / total_samples * 100:.2f}%)")
    else:
        print("Task: clustering / unlabeled data.")

    print("=" * 40)
    print("Run configuration:")
    print(f"  Dataset: {args.dataset_file}")
    print(f"  Data shape: {args.X_shape}")
    print(f"  Feature dimension: {args.X_shape[1]}")
    print(f"  Label distribution: normal={int(np.sum(labels == 0))}, anomalies={n_anomalies}")
    print(f"  Ball-entropy weighting: {'yes' if args.use_ball_entropy_weighting else 'no'}")
    print(f"  BIC threshold: {args.oneclass_bic_threshold}   log-likelihood gain tolerance: {args.oneclass_tau_ll}")
    print(f"  Random seed: {args.seed}")
    print("=" * 40)

    model = ProbabilisticGranularBallSystem(
        bic_threshold=args.oneclass_bic_threshold,
        tau_ll=args.oneclass_tau_ll,
        use_ball_entropy_weighting=args.use_ball_entropy_weighting,
        random_state=args.seed,
    )

    # --- Training and scoring ---
    if args.use_ball_entropy_weighting:
        balls, all_anomaly_scores, layer_weights = model.adaptive_split_evolution(
            X_train, max_iterations=args.max_iterations, eval_data=X_eval)

        print("\n=== Entropy-weighted accumulation of the per-level anomaly scores ===")
        print(f"{len(all_anomaly_scores)} levels, weights: {[f'{w:.4f}' for w in layer_weights]}")

        anomaly_scores = np.zeros_like(all_anomaly_scores[0])
        for scores, weight in zip(all_anomaly_scores, layer_weights):
            anomaly_scores += scores * weight
    else:
        balls, _, _ = model.adaptive_split_evolution(
            X_train, max_iterations=args.max_iterations, eval_data=X_eval)
        anomaly_scores = model.compute_anomaly_scores(X_eval)
        print("Using the final-level anomaly scores only.")

    # --- Evaluation ---
    results = None
    final_anomaly_scores = anomaly_scores

    if is_anomaly_detection_task:
        results = evaluate_anomaly_detection(final_anomaly_scores, labels_eval)
        print("\nEvaluation:")
        print(f"  AUC: {results['roc_auc']:.4f}")
        for key, value in results.items():
            if key not in ['best_threshold', 'roc_auc']:
                print(f"  {key}: {value:.4f}")

        y_pred = (final_anomaly_scores > results['best_threshold']).astype(int)
        cm = confusion_matrix(labels_eval, y_pred)
        print("Confusion matrix:")
        print("                pred:normal   pred:anomaly")
        print(f"  true:normal   {cm[0, 0]:8d}   {cm[0, 1]:8d}")
        print(f"  true:anomaly  {cm[1, 0]:8d}   {cm[1, 1]:8d}")

        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        print(f"False positive rate (FPR): {fpr:.4f} ({fp}/{fp + tn})")
        print(f"False negative rate (FNR): {fnr:.4f} ({fn}/{fn + tp})")

    return balls, final_anomaly_scores, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run HPGR for tabular one-class classification / anomaly detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', type=str, default=None, help='config file path (default: configs/default.yaml)')
    parser.add_argument('--dataset', type=str, required=True, help='dataset file name (.mat)')
    parser.add_argument('--use-ball-entropy-weighting', dest='use_ball_entropy_weighting',
                        action='store_true', default=None,
                        help="anomaly scoring: True = entropy-weighted accumulation over all levels; False = final level only")
    parser.add_argument('--oneclass-bic-threshold', type=float, default=None, help='local BIC split threshold')
    parser.add_argument('--oneclass-tau-ll', type=float, default=None, help='log-likelihood gain tolerance')
    parser.add_argument('--max-iterations', type=int, default=None, help='maximum number of splitting iterations')
    parser.add_argument('--seed', type=int, default=None, help='random seed')

    args = parser.parse_args()

    cfg = load_config(args.config)
    mcfg, dcfg = cfg['model'], cfg['data']

    # Anything not given on the command line falls back to configs/default.yaml.
    args.use_ball_entropy_weighting = (mcfg['use_ball_entropy_weighting']
                                       if args.use_ball_entropy_weighting is None else args.use_ball_entropy_weighting)
    args.oneclass_bic_threshold = (mcfg['bic_threshold']
                                   if args.oneclass_bic_threshold is None else args.oneclass_bic_threshold)
    args.oneclass_tau_ll = mcfg['tau_ll'] if args.oneclass_tau_ll is None else args.oneclass_tau_ll
    args.max_iterations = mcfg['max_iterations'] if args.max_iterations is None else args.max_iterations
    args.seed = mcfg['random_state'] if args.seed is None else args.seed

    dataset_dir = dcfg['dir'] if os.path.isabs(dcfg['dir']) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), dcfg['dir'])

    dataset_file = args.dataset if args.dataset.endswith('.mat') else args.dataset + '.mat'

    X, labels = load_dataset(dataset_dir, dataset_file)
    scaler = StandardScaler()
    scaler.fit(X[labels == 0])          # fit on normal samples only
    X_trans = scaler.transform(X)

    X_train = X_trans[labels == 0]      # one-class training uses normal samples only
    X_eval = X_trans                    # evaluation uses all samples

    args.dataset_file = dataset_file
    args.X_shape = X.shape

    balls, final_anomaly_scores, results = main(args, X_train, X_eval, labels)
    print("\nDone.")
