"""
Executable Runner Script for Stage 5 Evaluation.
Runs the evaluation audit pipeline and prints a detailed terminal summary.
"""

from .evaluator import ScenarioEvaluator


def main():
    evaluator = ScenarioEvaluator()
    report = evaluator.run_evaluation()

    summary = report["evaluation_summary"]
    stress_dist = report["stress_distribution"]
    div = report["diversity_analysis"]

    print("=" * 60)
    print("STAGE 5 EVALUATION ENGINEER AUDIT COMPLETE")
    print("=" * 60)
    print(f"Total Scenarios Audited : {summary['total_scenarios']}")
    print(f"Passed Scenarios        : {summary['passed']}")
    print(f"Review Scenarios        : {summary['review']}")
    print(f"Failed Scenarios        : {summary['failed']}")
    print(f"Average Stress Score    : {summary['average_stress_score']} / 5.0")
    print(f"Average Realism Score   : {summary['average_realism_score']} / 5.0")
    print(f"Blind Spots Covered     : {summary['blind_spot_coverage']} unique blind spots")
    print(f"Duplicate Scenarios     : {summary['duplicate_count']}")
    print(f"Strong Stress Cases     : {summary['strong_stress_cases']}")
    print(f"Extreme Stress Cases    : {summary['extreme_stress_cases']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
