from __future__ import annotations

from pathlib import Path


# Change this to your dataset YAML when you start training.
DATA_YAML = Path(r"F:\CollegeStudy\GitHub\deep-learning-group-project\datasets\Text Detection.v9i.yolov8\data.yaml")

# Pick any Ultralytics YOLOv8 pretrained checkpoint you want to fine-tune.
PRETRAINED_WEIGHTS = "yolov8n.pt"

# Basic training parameters.
EPOCHS = 100
IMGSZ = 640
BATCH = 16
DEVICE = 0

# Output folder under the current project.
PROJECT_DIR = Path("runs") / "detect"
RUN_NAME = "text_blackboard_detector"


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError(
            "Ultralytics is not installed. Run `pip install ultralytics` first."
        ) from e

    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    print("YOLOv8 training configuration")
    print(f"data.yaml: {DATA_YAML.resolve()}")
    print(f"pretrained weights: {PRETRAINED_WEIGHTS}")
    print(f"epochs: {EPOCHS}")
    print(f"imgsz: {IMGSZ}")
    print(f"batch: {BATCH}")
    print(f"device: {DEVICE}")
    print(f"project dir: {PROJECT_DIR}")
    print(f"run name: {RUN_NAME}")

    model = YOLO(PRETRAINED_WEIGHTS)
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        pretrained=True,
    )

    save_dir = Path(getattr(results, "save_dir", PROJECT_DIR / RUN_NAME))
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"

    print("\nTraining finished.")
    print(f"Run directory: {save_dir.resolve()}")
    print(f"Best weights: {best_weights.resolve()}")
    print(f"Last weights: {last_weights.resolve()}")


if __name__ == "__main__":
    main()
