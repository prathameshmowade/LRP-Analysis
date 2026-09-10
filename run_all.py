"""
Main Runner: Layer-wise Relevance Propagation (LRP) Analysis
============================================================
Executes both:
  1. Tabular LRP Analysis (Breast Cancer MLP)
  2. Visual LRP Analysis (Handwritten Digits CNN)

Generates full diagnostic prints, conservation law verifications,
and saves all plots to `output_plots/`.
"""

import os
import sys
import time
from tabular_lrp import run_tabular_analysis
from image_lrp import run_image_analysis


def main():
    print("=" * 70)
    print("    LAYER-WISE RELEVANCE PROPAGATION (LRP) ANALYSIS SUITE")
    print("    SEM 7 - Explainable AI (XAI) - Term Assignment Evaluation")
    print("=" * 70)

    start_time = time.time()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_plots")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Tabular Analysis
    run_tabular_analysis(output_dir=output_dir)

    # 2. Visual Image Analysis
    run_image_analysis(output_dir=output_dir)

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"  All LRP analyses completed successfully in {elapsed:.2f} seconds!")
    print(f"  Generated plots are stored in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
