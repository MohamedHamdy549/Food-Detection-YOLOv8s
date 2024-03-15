"""# Load YOLOv8s, train it on mnist160 for 3 epochs, and predict an image with it
from ultralytics import YOLO

# Initialize the YOLOv8s model with the yolov8s.pt weights
model = YOLO('yolov8s.pt')

# Train the model on the specified dataset for 3 epochs
model.train(data='datasets/food-v2i-yolov8-obb', epochs=10)

# Save the trained weights
model.save('weights/yolov8s_trained.pt')

# Perform inference on an image using the trained model
results = model('inference/images/img0.jpg')

# Display prediction results
results.show()"""
# Load YOLOv8s, train it on mnist160 for 3 epochs, and predict an image with it
from ultralytics import YOLO

# Initialize the YOLOv8s model with the yolov8s.pt weights
model = YOLO('yolov8s.pt')

# Specify the path to the YAML configuration file
yaml_config_path = 'datasets/food-v2i-yolov8-obb/data.yaml'

# Train the model using the configuration file
model.train(data=yaml_config_path, epochs=10)

# Save the trained weights
model.save('weights/yolov8s_trained.pt')

# Perform inference on an image using the trained model
results = model('inference/images/img0.jpg')

# Display prediction results
results.show()

