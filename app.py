from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import time
import threading

model = YOLO("model/V2_YOLOv8s.pt")

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

def process_image(image_data):
    try:
        img_data = base64.b64decode(image_data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"status": "error", "message": "Gagal decode gambar"}

        start_time = time.time()
        
        results = model(frame, conf=0.5)
        
        end_time = time.time()

        detections = []
        annotated_frame = frame.copy()

        for result in results:
            boxes = result.boxes.xyxy.numpy()
            confidences = result.boxes.conf.numpy()
            class_ids = result.boxes.cls.numpy().astype(int)
            class_names = model.names

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = f"{class_names[class_id]} {confidence:.2f}"
                color = (0, 255, 0)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                detections.append({
                    "class": class_names[class_id],
                    "confidence": float(confidence),
                    "box": [x1, y1, x2, y2]
                })

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        annotated_image = base64.b64encode(buffer).decode('utf-8')

        return {
            "status": "success",
            "detections": detections,
            "time": (end_time - start_time) * 1000,  # Time in ms
            "annotated_image": annotated_image
        }
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        image_data = data['image']

        result_holder = {}
        thread = threading.Thread(target=lambda: result_holder.update(process_image(image_data)))
        thread.start()
        thread.join()

        return jsonify(result_holder)
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
