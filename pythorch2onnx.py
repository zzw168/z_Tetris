from ultralytics import YOLO

# Load a model
model = YOLO('./yolov8n.pt')  # load an official model
# model = YOLO('./best.pt')  # load a custom trained

# Export the model
model.export(format='onnx')
