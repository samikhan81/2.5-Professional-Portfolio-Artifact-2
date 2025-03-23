import cv2
import torch
from pyttsx3 import init

# Load YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize text-to-speech engine
engine = init()

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform inference
    results = model(frame)
    
    # Parse results
    labels = results.pandas().xyxy[0]['name'].tolist()
    
    # Alert if specific objects are detected (e.g., "cell phone")
    if "cell phone" in labels:
        engine.say("Cell phone detected!")
        engine.runAndWait()
    
    # Display bounding boxes
    cv2.imshow('Live Object Detection', np.squeeze(results.render()))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()