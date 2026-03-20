import argparse

from config import MASKS_DIR, MODELS_DIR, PREDICTIONS_DIR
from inference.predict import run_inference
from training.train import train_model
from utils.metrics import evaluate_segmentation


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="AAA Segmentation using Hybrid Residual Attention U-Net + Swin Transformer",
	)
	mode_group = parser.add_mutually_exclusive_group(required=True)
	mode_group.add_argument("--train", action="store_true", help="Train the segmentation model")
	mode_group.add_argument("--predict", action="store_true", help="Run inference on new CT images")
	mode_group.add_argument("--evaluate", action="store_true", help="Evaluate saved predictions against ground-truth masks")

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
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if args.train:
		train_model()
		return

	if args.predict:
		if args.input is None:
			raise ValueError("--input is required when using --predict")
		saved_files = run_inference(input_path=args.input, model_path=args.model_path)
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


if __name__ == "__main__":
	main()

