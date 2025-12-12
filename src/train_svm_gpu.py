#!/usr/bin/env python3
"""
Train/Eval SVM on GPU using cuML SVC (RBF).
Usage:
  python train_svm_gpu.py features 8192 0 0 10 0.000122 1e-3
Args:
  prefix         : feature prefix (files: <prefix>_train_features.bin, ...)
  dim            : feature dimension (8192)
  sample_train   : 0 = full train, >0 = limit
  sample_test    : 0 = full test, >0 = limit
  C              : soft-margin C
  gamma          : RBF gamma; if <=0 uses 1/dim
  tol            : stopping tolerance (default 1e-3)
"""

import argparse
import struct
import time
from pathlib import Path

import numpy as np
from thundersvm import SVC


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("prefix", type=str, help="feature prefix")
    p.add_argument("dim", type=int, help="feature dimension (8192)")
    p.add_argument("sample_train", type=int, nargs="?", default=0)
    p.add_argument("sample_test", type=int, nargs="?", default=0)
    p.add_argument("C", type=float, nargs="?", default=10.0)
    p.add_argument("gamma", type=float, nargs="?", default=0.0)
    p.add_argument("tol", type=float, nargs="?", default=1e-3)
    return p.parse_args()


def load_split(prefix: str, split: str, dim: int, limit: int):
    feat_path = Path(f"{prefix}_{split}_features.bin")
    lbl_path = Path(f"{prefix}_{split}_labels.bin")
    feat = np.fromfile(feat_path, dtype=np.float32)
    lbl = np.fromfile(lbl_path, dtype=np.uint8)
    total = feat.size // dim
    if lbl.size != total:
        raise RuntimeError("Feature/label count mismatch")
    if limit > 0 and limit < total:
        total = limit
    feat = feat[: total * dim].reshape(total, dim)
    lbl = lbl[:total]
    return feat, lbl


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def main():
    args = parse_args()
    gamma = args.gamma if args.gamma > 0 else 1.0 / float(args.dim)

    t0 = time.perf_counter()
    X_train, y_train = load_split(args.prefix, "train", args.dim, args.sample_train)
    X_test, y_test = load_split(args.prefix, "test", args.dim, args.sample_test)
    t1 = time.perf_counter()
    print(f"Loaded train: {X_train.shape[0]}, test: {X_test.shape[0]}. Load time: {t1 - t0:.4f}s")

    svc = SVC(kernel="rbf", C=args.C, gamma=gamma, tol=args.tol, max_iter=-1)
    print(f"Training ThunderSVM SVC (RBF) C={args.C}, gamma={gamma}, tol={args.tol}")
    t_train0 = time.perf_counter()
    svc.fit(X_train, y_train)
    t_train1 = time.perf_counter()

    # Predict
    y_train_pred = svc.predict(X_train)
    y_test_pred = svc.predict(X_test)
    t_eval = time.perf_counter()

    # Accuracy
    train_acc = (y_train_pred == y_train).mean()
    test_acc = (y_test_pred == y_test).mean()
    print(f"Train accuracy: {train_acc * 100:.2f}% ({(y_train_pred == y_train).sum()}/{y_train.size})")
    print(f"Test  accuracy: {test_acc * 100:.2f}% ({(y_test_pred == y_test).sum()}/{y_test.size})")

    # Confusion matrices
    print("Train confusion matrix (rows=true, cols=pred):")
    cm_train = confusion_matrix(y_train, y_train_pred)
    for r in cm_train:
        print(" ".join(str(x) for x in r))
    print("Test confusion matrix (rows=true, cols=pred):")
    cm_test = confusion_matrix(y_test, y_test_pred)
    for r in cm_test:
        print(" ".join(str(x) for x in r))

    print(f"Timing: load={t1 - t0:.4f}s, train={t_train1 - t_train0:.4f}s, eval={t_eval - t_train1:.4f}s")


if __name__ == "__main__":
    main()
