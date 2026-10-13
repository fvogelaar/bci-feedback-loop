"""
Closed-loop real-time feedback pipeline
"""

from pylsl import StreamInlet, resolve_stream
from src.feedback import VisualBar
from src.controller import Controller

streams = resolve_stream("type", "BCI_PREDICTION")
inlet = StreamInlet(streams[0])

feedback = VisualBar()
controller = Controller()

while True:
    pred, _ = inlet.pull_sample()
    value = controller.update(pred[0])
    feedback.update(value)
