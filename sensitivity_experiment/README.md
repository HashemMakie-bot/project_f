# Project3 sensitivity experiment

Select a decision threshold using validation predictions only; no test file is read.

```bash
python sensitivity_experiment/analyze.py /path/to/validation_predictions.csv --out /path/to/validation_threshold.json --target 0.95
```

For the existing run (134 nodules from 70 patients), threshold 0.6535692811 gave 34/35 sensitivity and 79/99 specificity on the same validation data used to choose it. The prior 0.6964746714 threshold gave 32/35 and 80/99. These are development estimates, not proof of improvement on unseen data. Bootstrap intervals use resampling by patient and are conditional on the already-selected threshold; they do not remove selection bias.

The existing test set has been examined and is not a fresh holdout. Do not select or adjust thresholds using its results. Freeze the selected threshold and model hashes before a new independent evaluation. Model scores are not calibrated cancer probabilities. This changes the decision threshold, not model weights or full-volume detection performance.
