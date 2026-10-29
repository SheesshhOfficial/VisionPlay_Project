"""Command-line trainer for VisionPlay's landmark MLP (no webcam required)."""
import argparse
from pathlib import Path
import sys

from neural_learning import train_neural


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Train/evaluate a multi-layer landmark neural classifier.')
    parser.add_argument('--data', default=str(root / 'data' / 'samples.csv'))
    parser.add_argument('--model', default=str(root / 'data' / 'neural_model.npz'))
    parser.add_argument('--report', default=str(root / 'data' / 'neural_evaluation.json'))
    args = parser.parse_args(argv)
    try:
        _, report = train_neural(args.data, args.model, args.report)
    except (ValueError, OSError) as exc:
        print(f'Neural training unavailable: {exc}', file=sys.stderr)
        return 2
    print(f"Held-out session {report['test_session']}: accuracy {report['accuracy']:.1%} ({report['test_samples']} samples)")
    print(f"Model: {args.model}")
    if report.get('report_sidecar_error'):
        print(f"Report sidecar could not be written: {report['report_sidecar_error']}")
        print('The authoritative evaluation is embedded in the model artifact.')
    else:
        print(f"Report: {args.report}")
    print('Warning: personal dataset score is not general accuracy; see report limitations.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
