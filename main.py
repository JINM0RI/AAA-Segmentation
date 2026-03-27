import argparse
from pathlib import Path

from config import MASKS_DIR, MODELS_DIR, PLOTS_DIR, PREDICTIONS_DIR, POSTPROCESS_KERNEL_SIZE, PRED_THRESHOLD
from inference.predict import run_inference
from training.train import train_model
from utils.metrics import evaluate_segmentation, plot_mask_comparison_accuracy


def _build_thresholds(start: float, end: float, step: float) -> list[float]:
	if step <= 0:
		raise ValueError("--th-step must be > 0")
	if end < start:
		raise ValueError("--th-max must be >= --th-min")

	values: list[float] = []
	current = float(start)
	while current <= end + 1e-9:
		values.append(round(current, 4))
		current += step
	return values


def _run_threshold_sweep(args: argparse.Namespace) -> None:
	if args.input is None:
		raise ValueError("--input is required when using --sweep-thresholds")

	kernel = (int(args.kernel_size), int(args.kernel_size))
	thresholds = _build_thresholds(args.th_min, args.th_max, args.th_step)

	base_dir = Path(args.pred_dir) / "sweeps"
	base_dir.mkdir(parents=True, exist_ok=True)

	results: list[dict[str, float]] = []
	total = len(thresholds)
	for idx, th in enumerate(thresholds, start=1):
		print(f"Running threshold {idx}/{total}: {th:.3f}")
		run_dir = base_dir / f"th_{int(round(th * 1000)):04d}"
		run_dir.mkdir(parents=True, exist_ok=True)

		run_inference(
			input_path=args.input,
			model_path=args.model_path,
			output_dir=run_dir,
			threshold=th,
			kernel_size=kernel,
		)
		metrics = evaluate_segmentation(pred_dir=run_dir, mask_dir=args.mask_dir)
		print(
			f"  done: dice={metrics['dice']:.4f}, iou={metrics['iou']:.4f}, "
			f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}"
		)
		results.append(
			{
				"threshold": th,
				"dice": metrics["dice"],
				"iou": metrics["iou"],
				"precision": metrics["precision"],
				"recall": metrics["recall"],
			}
		)

	if not results:
		raise RuntimeError("Threshold sweep did not produce any results.")

	candidates = [r for r in results if r["precision"] >= float(args.target_precision)]
	if candidates:
		best = max(candidates, key=lambda r: (r["dice"], r["precision"], r["recall"]))
		selection_note = f"Best with precision >= {float(args.target_precision):.3f}"
	else:
		best = max(results, key=lambda r: (r["dice"], r["precision"], r["recall"]))
		selection_note = (
			f"No threshold reached precision target {float(args.target_precision):.3f}; "
			"showing best Dice instead"
		)

	print("\n## Threshold Sweep Results\n")
	print("threshold | dice   | iou    | precision | recall")
	for r in results:
		print(
			f"{r['threshold']:.3f}     | "
			f"{r['dice']:.4f} | {r['iou']:.4f} | {r['precision']:.4f}   | {r['recall']:.4f}"
		)

	print("\n## Best Threshold\n")
	print(selection_note)
	print(f"Threshold: {best['threshold']:.3f}")
	print(f"Dice: {best['dice']:.4f}")
	print(f"IoU: {best['iou']:.4f}")
	print(f"Precision: {best['precision']:.4f}")
	print(f"Recall: {best['recall']:.4f}")
	print(f"Predictions saved under: {base_dir}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="AAA Segmentation using Hybrid Residual Attention U-Net + Swin Transformer",
	)
	mode_group = parser.add_mutually_exclusive_group(required=True)
	mode_group.add_argument("--train", action="store_true", help="Train the segmentation model")
	mode_group.add_argument("--predict", action="store_true", help="Run inference on new CT images")
	mode_group.add_argument("--evaluate", action="store_true", help="Evaluate saved predictions against ground-truth masks")
	mode_group.add_argument(
		"--sweep-thresholds",
		action="store_true",
		help="Run inference/evaluation across a threshold range and report best threshold",
	)

	parser.add_argument(
		"--input",
		type=str,
		default=None,
		help="Input image file or folder for prediction mode",
	)
	parser.add_argument(
		"--model-path",
		type=str,
		default=str(MODELS_DIR / "best_model.pth"),
		help="Path to trained model weights",
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default=PRED_THRESHOLD,
		help="Prediction threshold for converting probability map to binary mask",
	)
	parser.add_argument(
		"--kernel-size",
		type=int,
		default=POSTPROCESS_KERNEL_SIZE[0],
		help="Morphology kernel size used in post-processing",
	)
	parser.add_argument("--th-min", type=float, default=0.20, help="Minimum threshold for sweep mode")
	parser.add_argument("--th-max", type=float, default=0.70, help="Maximum threshold for sweep mode")
	parser.add_argument("--th-step", type=float, default=0.05, help="Threshold step for sweep mode")
	parser.add_argument(
		"--target-precision",
		type=float,
		default=0.95,
		help="Precision target used to choose best threshold in sweep mode",
	)
	parser.add_argument(
		"--pred-dir",
		type=str,
		default=str(PREDICTIONS_DIR),
		help="Directory containing saved predicted masks for evaluation",
	)
	parser.add_argument(
		"--mask-dir",
		type=str,
		default=str(MASKS_DIR),
		help="Directory containing ground-truth masks for evaluation",
	)
	parser.add_argument(
		"--plot-compare",
		action="store_true",
		help="Generate Ground Truth vs Predicted mask comparison graph",
	)
	parser.add_argument(
		"--plot-out",
		type=str,
		default=str(PLOTS_DIR / "mask_accuracy_comparison.png"),
		help="Output PNG path for comparison graph",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if args.train:
		train_model()
		return

	if args.predict:
		if args.input is None:
			raise ValueError("--input is required when using --predict")
		kernel = (int(args.kernel_size), int(args.kernel_size))
		saved_files = run_inference(
			input_path=args.input,
			model_path=args.model_path,
			threshold=float(args.threshold),
			kernel_size=kernel,
		)
		print("Saved predictions:")
		for path in saved_files:
			print(f" - {path}")
		return

	if args.evaluate:
		eval_metrics = evaluate_segmentation(pred_dir=args.pred_dir, mask_dir=args.mask_dir)
		print("\n## Evaluation Summary\n")
		print(f"Dice Score: {eval_metrics['dice']:.4f}")
		print(f"IoU Score: {eval_metrics['iou']:.4f}")
		print(f"Precision: {eval_metrics['precision']:.4f}")
		print(f"Recall: {eval_metrics['recall']:.4f}")

		if args.plot_compare:
			plot_info = plot_mask_comparison_accuracy(
				pred_dir=args.pred_dir,
				mask_dir=args.mask_dir,
				output_path=args.plot_out,
			)
			print("\n## Comparison Plot\n")
			print(f"Mean Ground Truth Ratio: {plot_info['mean_ground_truth']:.4f}")
			print(f"Mean Predicted Ratio: {plot_info['mean_predicted']:.4f}")
			print(f"Images Plotted: {int(plot_info['num_images'])}")
			print(f"Saved plot: {plot_info['plot_path']}")
		return

	if args.sweep_thresholds:
		_run_threshold_sweep(args)


if __name__ == "__main__":
	main()

