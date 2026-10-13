"""
Real-time classification pipeline
"""

import time
from pylsl import StreamInlet, resolve_stream
from src.classifier import OnlineLDA
from src.metrics import OnlineMetrics
from examples.train_offline import offline_trainer

def online_training():
    streams = resolve_stream("type", "EEG_FEATURES")
    inlet = StreamInlet(streams[0])

    cal_model = offline_trainer()     # this runs the offline training with calibration file
    metrics = OnlineMetrics()


    while True:
        t0 = time.time()
        feature, _ = inlet.pull_sample()

        y = cal_model.predict(feature)
        if y is not None:
            metrics.log_latency(t0)
            print("Prediction:", y,
                "Mean latency:", metrics.mean_latency())
