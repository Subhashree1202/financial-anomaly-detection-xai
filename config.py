# ─────────────────────────────────────────────
#  Project-wide configuration
# ─────────────────────────────────────────────

TICKER      = "^GSPC"          # S&P 500 index
START_DATE  = "2005-01-01"
END_DATE    = "2024-12-31"

# Sliding window fed into the LSTM autoencoder
WINDOW_SIZE = 30               # trading days (~6 weeks)

# Train / validation split (no look-ahead)
TRAIN_END   = "2019-12-31"

# Model
HIDDEN_DIM  = 64
LATENT_DIM  = 16
N_LAYERS    = 2
DROPOUT     = 0.2
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 1e-3
SEED        = 42

# Anomaly threshold: flag windows whose reconstruction
# error exceeds THRESH_SIGMA standard deviations above
# the training-set mean error
THRESH_SIGMA = 3.0

# Known market stress events for annotation
KNOWN_EVENTS = {
    "2008-09-15": "Lehman collapse",
    "2010-05-06": "Flash Crash",
    "2015-08-24": "China selloff",
    "2018-12-24": "Q4 2018 selloff",
    "2020-03-16": "COVID crash",
    "2022-06-13": "Rate-hike selloff",
}
