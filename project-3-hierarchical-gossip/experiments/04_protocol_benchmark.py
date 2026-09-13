"""Run the reproducible local/FedAvg/MH/oracle CIFAR-100 completion benchmark."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.protocol_benchmark import main


if __name__ == "__main__":
    main()
