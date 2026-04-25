"""
Statistical Testing Module (Phase 5-B)
========================================

Provides statistical tests for pairwise controller comparison,
effect size computation, and confidence intervals.

Implements:
    - Wilcoxon signed-rank test (non-parametric paired comparison)
    - Cohen's d (effect size for continuous metrics)
    - Wilson confidence interval (for collision proportions)
    - Bonferroni correction (multiple comparison adjustment)

Reference:
    Risk-Aware Hybrid LQR-MPC Navigation, Phase 5 - Evaluation Overhaul
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats


@dataclass
class PairwiseResult:
    """Result of a pairwise statistical comparison."""
    controller_a: str
    controller_b: str
    metric: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    p_value: float
    effect_size: float  # Cohen's d
    significant: bool   # After Bonferroni correction
    direction: str      # 'a_better', 'b_better', or 'no_difference'


@dataclass
class ProportionCI:
    """Wilson confidence interval for a proportion."""
    proportion: float
    lower: float
    upper: float
    n_success: int
    n_total: int
    alpha: float


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two samples.
    
    Cohen's d = (mean_a - mean_b) / pooled_std
    
    Interpretation:
        |d| < 0.2: negligible
        0.2 ≤ |d| < 0.5: small
        0.5 ≤ |d| < 0.8: medium
        |d| ≥ 0.8: large
    
    Args:
        a, b: Sample arrays
        
    Returns:
        Cohen's d value (positive means a > b)
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    
    mean_diff = np.mean(a) - np.mean(b)
    
    # Pooled standard deviation
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    
    if pooled_std < 1e-12:
        return 0.0
    
    return mean_diff / pooled_std


def wilson_ci(n_success: int, n_total: int, alpha: float = 0.05) -> ProportionCI:
    """
    Compute Wilson score confidence interval for a proportion.
    
    More accurate than the normal approximation for small samples
    and extreme proportions (near 0 or 1).
    
    Args:
        n_success: Number of successes (e.g., collisions)
        n_total: Total number of trials
        alpha: Significance level (default 0.05 for 95% CI)
        
    Returns:
        ProportionCI with proportion, lower, upper bounds
    """
    if n_total == 0:
        return ProportionCI(0.0, 0.0, 1.0, 0, 0, alpha)
    
    p_hat = n_success / n_total
    z = stats.norm.ppf(1 - alpha / 2)
    
    denominator = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total) / denominator
    
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    
    return ProportionCI(
        proportion=p_hat,
        lower=lower,
        upper=upper,
        n_success=n_success,
        n_total=n_total,
        alpha=alpha
    )


def wilcoxon_pairwise(results_dict: Dict[str, np.ndarray], 
                       metric: str,
                       alpha: float = 0.05) -> List[PairwiseResult]:
    """
    Perform pairwise Wilcoxon signed-rank tests across all controller pairs.
    
    The Wilcoxon signed-rank test is a non-parametric alternative to the
    paired t-test. It does not assume normality of the differences.
    
    Applies Bonferroni correction for multiple comparisons.
    
    Args:
        results_dict: Dict mapping controller names to metric arrays.
                      E.g., {'lqr': np.array([...]), 'mpc': np.array([...]), ...}
        metric: Name of the metric being compared (for reporting)
        alpha: Base significance level (before correction)
        
    Returns:
        List of PairwiseResult for each pair
    """
    controllers = sorted(results_dict.keys())
    n_pairs = len(controllers) * (len(controllers) - 1) // 2
    
    # Bonferroni-corrected alpha
    alpha_corrected = alpha / max(n_pairs, 1)
    
    results = []
    
    for i in range(len(controllers)):
        for j in range(i + 1, len(controllers)):
            name_a = controllers[i]
            name_b = controllers[j]
            data_a = np.asarray(results_dict[name_a], dtype=float)
            data_b = np.asarray(results_dict[name_b], dtype=float)
            
            # Ensure same length (paired test)
            min_len = min(len(data_a), len(data_b))
            data_a = data_a[:min_len]
            data_b = data_b[:min_len]
            
            if min_len < 5:
                # Too few samples for meaningful test
                results.append(PairwiseResult(
                    controller_a=name_a,
                    controller_b=name_b,
                    metric=metric,
                    mean_a=float(np.mean(data_a)) if len(data_a) > 0 else 0.0,
                    mean_b=float(np.mean(data_b)) if len(data_b) > 0 else 0.0,
                    std_a=float(np.std(data_a, ddof=1)) if len(data_a) > 1 else 0.0,
                    std_b=float(np.std(data_b, ddof=1)) if len(data_b) > 1 else 0.0,
                    p_value=1.0,
                    effect_size=0.0,
                    significant=False,
                    direction='insufficient_data'
                ))
                continue
            
            # Check if arrays are identical (zero variance in differences)
            diff = data_a - data_b
            if np.all(np.abs(diff) < 1e-12):
                p_value = 1.0
            else:
                try:
                    stat, p_value = stats.wilcoxon(data_a, data_b, alternative='two-sided')
                except ValueError:
                    # Can happen if all differences are zero
                    p_value = 1.0
            
            # Effect size
            d = cohen_d(data_a, data_b)
            
            # Determine direction
            mean_a = float(np.mean(data_a))
            mean_b = float(np.mean(data_b))
            
            if p_value < alpha_corrected:
                direction = 'a_better' if mean_a < mean_b else 'b_better'
                # For metrics where higher is better (e.g., completion rate),
                # the caller should interpret direction accordingly
            else:
                direction = 'no_difference'
            
            results.append(PairwiseResult(
                controller_a=name_a,
                controller_b=name_b,
                metric=metric,
                mean_a=mean_a,
                mean_b=mean_b,
                std_a=float(np.std(data_a, ddof=1)),
                std_b=float(np.std(data_b, ddof=1)),
                p_value=p_value,
                effect_size=d,
                significant=p_value < alpha_corrected,
                direction=direction
            ))
    
    return results


def format_comparison_table(results: List[PairwiseResult]) -> str:
    """
    Format pairwise comparison results as a markdown table.
    
    Args:
        results: List of PairwiseResult objects
        
    Returns:
        Formatted markdown table string
    """
    if not results:
        return "No comparison results available."
    
    lines = []
    metric = results[0].metric
    lines.append(f"### Pairwise Comparison: {metric}")
    lines.append("")
    lines.append("| Controller A | Controller B | Mean A | Mean B | Cohen's d | p-value | Significant |")
    lines.append("|-------------|-------------|--------|--------|-----------|---------|-------------|")
    
    for r in results:
        sig_marker = "✓" if r.significant else "✗"
        lines.append(
            f"| {r.controller_a} | {r.controller_b} | "
            f"{r.mean_a:.4f} | {r.mean_b:.4f} | "
            f"{r.effect_size:.3f} | {r.p_value:.4f} | {sig_marker} |"
        )
    
    return "\n".join(lines)


def validate_results_stochastic(results_dict: Dict[str, np.ndarray]) -> bool:
    """
    P5-A validation: Assert at least one metric has nonzero standard deviation.
    
    Prevents saving deterministic (invalid) results from zero-noise evaluation.
    
    Args:
        results_dict: Dict mapping metric names to result arrays
        
    Returns:
        True if results have meaningful variance, False if deterministic
        
    Raises:
        ValueError: If all metrics have zero variance (deterministic results)
    """
    has_variance = False
    for metric_name, values in results_dict.items():
        arr = np.asarray(values, dtype=float)
        if len(arr) > 1 and np.std(arr) > 1e-10:
            has_variance = True
            break
    
    if not has_variance:
        raise ValueError(
            "P5-A VALIDATION FAILURE: All metrics have zero standard deviation. "
            "This indicates deterministic evaluation (likely noise_std=0.0). "
            "Enable stochastic noise (sigma_p >= 0.05) before saving results."
        )
    
    return True
