"""FaceAI - contactless health screening from facial video via rPPG."""

import os

__version__ = "0.1.0"

# Data governance (PDPA): no usage telemetry leaves this machine. The same values are
# set as conda env vars on `faceai`; setdefault here also covers runs that call the
# env's python by full path without activating. Names verified against the installed
# huggingface_hub 1.32.0, gradio 6.28.0 and mlflow 3.16.1 sources.
for _name, _value in (
    ("HF_HUB_DISABLE_TELEMETRY", "1"),
    ("GRADIO_ANALYTICS_ENABLED", "False"),
    ("MLFLOW_DISABLE_TELEMETRY", "true"),
    ("DO_NOT_TRACK", "true"),
):
    os.environ.setdefault(_name, _value)
