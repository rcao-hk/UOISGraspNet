from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data="uoais_sim.yaml", epochs=24, imgsz=640, batch=32, device=[1, 2])