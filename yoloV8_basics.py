from ultralytics import YOLO
import os

# load a pretrained YOLOv8n model
model = YOLO("weights/yolov8s_trained.pt", "v8")

# predict on an image
detection_output = model.predict(source="inference/images/img1.jpg", conf=0.25, save=True)

# dir to save output img
output_dir = 'inference/runs'
os.makedirs(output_dir, exist_ok=True)

# file name of output img
output_img = os.path.join(output_dir, "outImg.jpg")

# save
detection_output[0].save(output_img)
print("Output image saved at: ", output_img)

# Display tensor array
print(detection_output)

# Display numpy array
print(detection_output[0].numpy())
