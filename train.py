from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model

# Train the model (2200 images as per abstract)
results = model.train(
    data='data/indian_highway_animals.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='animal_detection_thesis'
)