import re

with open("docs/Work_Progress.md", "r") as f:
    content = f.read()

# Mark Phase 4 & 5 as Complete and add links
content = content.replace("### Phase 4: Stochastic Risk Supervisor (Advanced/Novelty) (Active)", "### Phase 4: Stochastic Risk Supervisor (Advanced/Novelty) \u2705 (Complete)")
content = content.replace("### Phase 5: Stochastic Experiments & Validation (Active)", "### Phase 5: Stochastic Experiments & Validation \u2705 (Complete)")
content = content.replace("[ ] Shift the underlying optimization to Stochastic Model Predictive Control (SMPC)", "[x] Shift the underlying optimization to Stochastic Model Predictive Control (SMPC)")
content = content.replace("[ ] Implement Mahalanobis distance Covariance overlap", "[x] Implement Mahalanobis distance Covariance overlap")
content = content.replace("[ ] Predict constraint violations under Gaussian disturbance", "[x] Predict constraint violations under Gaussian disturbance bounds automatically")
content = content.replace("[ ] Validate Mahalanobis-based SMPC", "[x] Validate Mahalanobis-based SMPC")
content = content.replace("[ ] Measure computational savings", "[x] Measure computational savings")
content = content.replace("[ ] Experimental analysis via tuning", "[x] Experimental analysis via tuning")

with open("docs/Work_Progress.md", "w") as f:
    f.write(content)
