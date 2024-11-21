"""
This code defines a **`PlayerTracker`** class that leverages the **YOLO (You Only Look Once)** object detection model to track players in a video, specifically identifying and drawing bounding boxes around players. The class performs several functions including player detection, filtering detections based on positions, and drawing the detected players in video frames. Let's break down the code in detail:

### **Imports and Initialization**
```python
from ultralytics import YOLO 
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox
```
- **`ultralytics.YOLO`**: This imports the YOLO model for object detection from the `ultralytics` package. YOLO is commonly used for real-time object detection tasks.
- **`cv2`**: The OpenCV library is used for image/video processing (like drawing bounding boxes).
- **`pickle`**: Used for saving and loading Python objects to/from files. It is used here to save player detection results.
- **`sys.path.append('../')`**: This adds the parent directory to the Python path, allowing importing modules from there.
- **`measure_distance` and `get_center_of_bbox`**: These utility functions are imported from a separate module, presumably for measuring distances between player locations and obtaining the center of a bounding box.

### **PlayerTracker Class**
The **`PlayerTracker`** class is responsible for detecting and tracking players in a video using the YOLO object detection model.

#### **`__init__` Method**
```python
def __init__(self, model_path):
    self.model = YOLO(model_path)
```
- This method initializes the tracker with a YOLO model loaded from the given `model_path`. The `YOLO` class is responsible for object detection, and the model is used to detect players in the frames.

#### **`choose_and_filter_players` Method**
```python
def choose_and_filter_players(self, court_keypoints, player_detections):
    player_detections_first_frame = player_detections[0]
    chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
    filtered_player_detections = []
    for player_dict in player_detections:
        filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
        filtered_player_detections.append(filtered_player_dict)
    return filtered_player_detections
```
- **Purpose**: This method filters the player detections across all frames based on their proximity to specific keypoints (likely corresponding to certain positions on a court).
- **Key Components**:
  - **`court_keypoints`**: This is a list of points on the court that are used to determine the location of players.
  - **`player_detections_first_frame`**: It takes the detections from the first frame to select players.
  - **`choose_players`**: It calls this method to decide which players are relevant based on their proximity to the court keypoints.
  - **Filtering**: Once the players are chosen, it filters the player detections to retain only those players that are relevant across all frames.

#### **`choose_players` Method**
```python
def choose_players(self, court_keypoints, player_dict):
    distances = []
    for track_id, bbox in player_dict.items():
        player_center = get_center_of_bbox(bbox)
        min_distance = float('inf')
        for i in range(0,len(court_keypoints),2):
            court_keypoint = (court_keypoints[i], court_keypoints[i+1])
            distance = measure_distance(player_center, court_keypoint)
            if distance < min_distance:
                min_distance = distance
        distances.append((track_id, min_distance))

    distances.sort(key = lambda x: x[1])
    chosen_players = [distances[0][0], distances[1][0]]
    return chosen_players
```
- **Purpose**: This method chooses the two closest players to specific court keypoints.
- **Key Components**:
  - **`player_dict`**: Contains player track IDs and their bounding boxes.
  - **`get_center_of_bbox`**: This function is used to get the center point of each player's bounding box.
  - **`measure_distance`**: Calculates the distance between the player’s center and each court keypoint. The closest players are selected.
  - **Sorting**: The players are sorted based on their distance to the court keypoints, and the two closest players are chosen.

#### **`detect_frames` Method**
```python
def detect_frames(self, frames, read_from_stub=False, stub_path=None):
    player_detections = []
    if read_from_stub and stub_path is not None:
        with open(stub_path, 'rb') as f:
            player_detections = pickle.load(f)
        return player_detections

    for frame in frames:
        player_dict = self.detect_frame(frame)
        player_detections.append(player_dict)

    if stub_path is not None:
        with open(stub_path, 'wb') as f:
            pickle.dump(player_detections, f)

    return player_detections
```
- **Purpose**: This method processes multiple video frames to detect players and return their bounding boxes.
- **Key Components**:
  - **`read_from_stub`**: If set to `True`, it loads player detections from a previously saved file (`stub_path`).
  - **`detect_frame`**: Calls this method for each frame to detect players.
  - **`pickle.dump`**: If `stub_path` is provided, it saves the player detections to a file for later use.

#### **`detect_frame` Method**
```python
def detect_frame(self, frame):
    results = self.model.track(frame, persist=True)[0]
    id_name_dict = results.names

    player_dict = {}
    for box in results.boxes:
        track_id = int(box.id.tolist()[0])
        result = box.xyxy.tolist()[0]
        object_cls_id = box.cls.tolist()[0]
        object_cls_name = id_name_dict[object_cls_id]
        if object_cls_name == "person":
            player_dict[track_id] = result

    return player_dict
```
- **Purpose**: Detects players in a single frame using the YOLO model.
- **Key Components**:
  - **`self.model.track`**: Applies the YOLO model to track objects (players) in the frame.
  - **`results.names`**: A dictionary mapping class IDs to object names.
  - **Bounding Boxes**: For each detected object, it checks if the object is a person (`"person"`) and then adds it to `player_dict` with its tracking ID and bounding box coordinates.

#### **`draw_bboxes` Method**
```python
def draw_bboxes(self, video_frames, player_detections):
    output_video_frames = []
    for frame, player_dict in zip(video_frames, player_detections):
        for track_id, bbox in player_dict.items():
            x1, y1, x2, y2 = bbox
            cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        output_video_frames.append(frame)
    
    return output_video_frames
```
- **Purpose**: This method draws bounding boxes and player IDs on the video frames.
- **Key Components**:
  - **`cv2.putText`**: Adds text (the player ID) on the frame near the player's bounding box.
  - **`cv2.rectangle`**: Draws the bounding box around the player.
  - **`output_video_frames`**: This list holds the frames after annotations (bounding boxes and player IDs).

### **Summary of the Workflow**
1. **Detection**: The model detects players in video frames, identifying them by their bounding boxes.
2. **Filtering**: The system chooses and filters relevant players based on their proximity to court keypoints.
3. **Tracking**: Players are tracked across frames, with bounding boxes drawn around them.
4. **Output**: Annotated frames are generated, showing player IDs and their positions.

This class provides a useful framework for tracking players in sports analytics, especially for real-time applications in video analysis."""
from ultralytics import YOLO 
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0,len(court_keypoints),2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
        
        # sorrt the distances in ascending order
        distances.sort(key = lambda x: x[1])
        # Choose the first 2 tracks
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players


    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self,frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
        
        return player_dict

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames


    