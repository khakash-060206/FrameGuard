"""FrameGuard facial manipulation detection prototype."""

# Scores within this many log-odds of the decision threshold are reported as "uncertain". Fixed a priori (not tuned
# on test data); on the Run 2 test set it flags 6.3% of videos, whose accuracy is 59.5% versus 97.1% for the rest.
# Kept here, free of ML imports, so the dashboard can show the same band without loading PyTorch.
UNCERTAIN_MARGIN = 1.5

