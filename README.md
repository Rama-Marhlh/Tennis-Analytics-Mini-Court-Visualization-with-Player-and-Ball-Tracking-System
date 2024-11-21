

# **Tennis Player and Ball Tracking with Mini-Court Visualization**

This project focuses on analyzing tennis match videos by tracking players, detecting ball movements, and predicting court keypoints to generate insightful analytics. The outputs include a mini-court representation, shot analysis, speed calculations, and interactive data visualizations.

---

## **Overview**
The project integrates advanced computer vision techniques with YOLO (You Only Look Once) models and a custom-trained neural network to achieve three main objectives:
1. **Player Tracking**: Identifying and assigning unique IDs to players using YOLOv8.
2. **Ball Detection**: Leveraging a custom-trained YOLOv5 model for precise tennis ball detection.
3. **Court Keypoint Prediction**: Detecting and mapping tennis court keypoints using a ResNet-50-based model.

The final outputs include visually enhanced videos, data for performance analysis, and a real-time mini-court visualization for shot and speed analysis.

---

## **Project Workflow**

### 1. **Player Tracking**
We utilize the YOLOv8 model from [Ultralytics](https://github.com/ultralytics/yolov8) to track players in the input video.

#### Code Snippet:
```python
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8x')

# Perform tracking on the input video
result = model.track('input_videos/input_video.mp4', conf=0.2, save=True)
```
- **Why YOLOv8?**  
  YOLOv8 excels in object detection and tracking, making it suitable for assigning unique IDs to players.
- **Output**: Each player in the video is assigned a unique ID for further analysis.

---

### 2. **Ball Detection**
While YOLOv8 performs well for player tracking, it struggles with accurately detecting the tennis ball. To address this, we trained a YOLOv5 model on a custom dataset.

#### Training Process:
- Dataset: [Tennis Ball Detection Dataset](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection/dataset/6)
- Training Notebook: `tennis_ball_detector_training.ipynb`
- Outputs:
  - `best.pt`: Model weights with the best validation performance.
  - `last.pt`: Model weights from the final epoch.
  - Stored in the `runs/train/` directory.

#### Running the Trained Model:
```python
from ultralytics import YOLO

# Load the YOLOv5 model trained on tennis ball data
model = YOLO('models/best.pt')

# Predict tennis ball locations in the video
result = model.predict('input_videos/input_video.mp4', conf=0.2, save=True)
```
- **Why Prediction Instead of Tracking?**  
  Since there's only one tennis ball to detect in each frame, tracking is unnecessary. Prediction offers precise localization.

---

### 3. **Court Keypoint Prediction**
Keypoint detection helps map the tennis court and its lines, aiding in the mini-court visualization. A custom model based on ResNet-50 is used for this task.

#### Training Details:
- Dataset: [Tennis Court Keypoints Dataset](https://github.com/yastrebksv/TennisCourtDetector)
- Base Model: Pretrained ResNet-50 from ImageNet.
- Customization:
  - Final layer outputs `(14 * 2)` values representing the `(x, y)` coordinates of 14 keypoints.
  - Loss Function: Mean Squared Error (MSE).
  - Optimizer: Adam with a learning rate of `1e-4`.
- Training Notebook: `tennis_court_keypoints.ipynb`

#### Running the Keypoint Detection Model:
```python
from court_keypoints import CourtLineDetector

# Load the court line detector model
court_model_path = "models/keypoints_model.pth"
court_line_detector = CourtLineDetector(court_model_path)

# Predict keypoints from video frames
court_keypoints = court_line_detector.predict(video_frames[0])
```
- **Output**: Detected keypoints are used to map the court lines and dimensions.

---

### 4. **Mini-Court Representation**
Using the tracked player positions, detected ball locations, and court keypoints, a mini-court visualization is created. This representation provides real-time insights into player and ball movements.

---

### 5. **Shot and Speed Analysis**
By analyzing the ball's position across frames:
- **Shot Type**: Identified based on ball trajectory and player positioning.
- **Speed**: Calculated using the distance traveled by the ball between frames and the frame rate of the video.

---

### 6. **Data Aggregation and Visualization**
The project aggregates the extracted data and presents it visually:
- Player positions, ball trajectories, and court mapping overlaid on the video.
- Statistical summaries of player movement and shot analysis.

---

## **Training and Output Files**
- **Trained Models:**
  - YOLOv5 (`best.pt`, `last.pt`) for tennis ball detection: [Google Drive Link](https://drive.google.com/drive/folders/1-LoCeWVH1-09TWE1-RT5IzRi3_D2DXsB?usp=sharing)
  - Court Keypoint Detection Model (`keypoints_model.pth`): [Google Drive Link](https://drive.google.com/drive/folders/1-LoCeWVH1-09TWE1-RT5IzRi3_D2DXsB?usp=sharing)
- **Output Videos:**  
  Processed videos after each step can be found [here](https://drive.google.com/drive/folders/1zj01fNp0wmAVr4f45sNybSXsvAFALkyl?usp=sharing).

---

## **How to Run**
1. Clone the repository.
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run `main.ipynb` to process the video and generate outputs.

---

## **Acknowledgments**
- **YOLO Models**: [Ultralytics](https://github.com/ultralytics)
- **Datasets**: 
  - [Tennis Ball Detection](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection/dataset/6)
  - [Tennis Court Keypoints](https://github.com/yastrebksv/TennisCourtDetector)

This project demonstrates the potential of computer vision in sports analytics, providing actionable insights into tennis match dynamics.

