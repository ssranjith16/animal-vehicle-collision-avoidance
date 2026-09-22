"""
Custom model training script for Indian highway animals
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dataset_yaml(data_dir: Path, output_path: Path):
    """Create YAML configuration for custom dataset"""
    
    # Define classes (Indian highway animals)
    classes = [
        'cow',
        'buffalo', 
        'dog',
        'nilgai',
        'camel',
        'elephant',
        'horse',
        'goat'
    ]
    
    dataset_config = {
        'path': str(data_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(classes),
        'names': classes
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    logger.info(f"Dataset configuration saved to {output_path}")
    return output_path

def train_model(data_yaml: str, model_name: str = 'yolov8n.pt', 
                epochs: int = 100, imgsz: int = 640, batch: int = 16):
    """
    Train custom animal detection model
    
    Args:
        data_yaml: Path to dataset YAML file
        model_name: Base model to use
        epochs: Number of training epochs
        imgsz: Image size
        batch: Batch size
    """
    logger.info("Starting model training...")
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Base model: {model_name}")
    logger.info(f"Epochs: {epochs}, Image size: {imgsz}, Batch size: {batch}")
    
    # Load model
    model = YOLO(model_name)
    
    # Train model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name='indian_animal_detector',
        patience=50,
        save=True,
        save_period=10,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=42,
        device='cuda',  # Use 'cpu' if no GPU
        workers=4,
        project='runs/train',
        exist_ok=True
    )
    
    logger.info("Training completed!")
    logger.info(f"Best model saved at: {results.save_dir}")
    
    # Evaluate model
    metrics = model.val()
    logger.info(f"Validation metrics: {metrics}")
    
    return model

def export_model(model_path: str, format: str = 'onnx'):
    """
    Export trained model to different formats
    
    Args:
        model_path: Path to trained model
        format: Export format ('onnx', 'tflite', 'torchscript')
    """
    model = YOLO(model_path)
    model.export(format=format)
    logger.info(f"Model exported to {format} format")

def main():
    parser = argparse.ArgumentParser(description='Train custom animal detection model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'],
                       help='Base model to use')
    parser.add_argument('--export', action='store_true',
                       help='Export model after training')
    
    args = parser.parse_args()
    
    # Create dataset YAML
    data_dir = Path(args.data)
    yaml_path = data_dir / 'dataset.yaml'
    create_dataset_yaml(data_dir, yaml_path)
    
    # Train model
    model = train_model(
        data_yaml=str(yaml_path),
        model_name=args.model,
        epochs=args.epochs,
        batch=args.batch
    )
    
    # Export if requested
    if args.export:
        export_model('runs/train/indian_animal_detector/weights/best.pt')

if __name__ == "__main__":
    main()