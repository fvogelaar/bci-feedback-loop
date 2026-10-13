"""
Real-time classification pipeline
"""

import time
from pylsl import StreamInlet, resolve_stream
from src.classifier import OnlineLDA
from src.metrics import OnlineMetrics

streams = resolve_stream("type", "EEG_FEATURES")
inlet = StreamInlet(streams[0])

clf = OnlineLDA()
metrics = OnlineMetrics()

# clf.train(X_train, y_train) must be called before running

while True:
    t0 = time.time()
    feature, _ = inlet.pull_sample()

    y = clf.predict(feature)
    if y is not None:
        metrics.log_latency(t0)
        print("Prediction:", y,
              "Mean latency:", metrics.mean_latency())
