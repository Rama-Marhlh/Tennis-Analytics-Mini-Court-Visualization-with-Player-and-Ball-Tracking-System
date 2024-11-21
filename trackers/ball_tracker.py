"""
### **Detailed Explanation of the Code**

The provided code is part of a **BallTracker** class that is designed to track and analyze the movement of the ball in a video of a sports game ( tennis) ,The class uses the **YOLO (You Only Look Once)** model from **Ultralytics** for object detection to track the ball's position across video frames and perform various analyses, such as detecting ball hits and interpolating missing ball positions. Below is a detailed explanation of each function and component.

---

### **Imports**
```python
from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
```
- **YOLO**: The Ultralytics YOLO model is used for object detection in the frames, specifically detecting the ball in each frame.
- **cv2**: OpenCV is used for video frame processing, drawing bounding boxes, and other computer vision tasks.
- **pickle**: Python’s serialization module is used to save and load the ball detection results between video frames.
- **pandas**: This library is used for data manipulation, particularly for handling ball position data in the form of a DataFrame and performing interpolation.

---

### **Class Initialization**
```python
class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
```
- **BallTracker**: This class is initialized with the path to a pre-trained YOLO model. This model is used to detect the ball in video frames.
- **`self.model`**: The YOLO model instance is created by passing the `model_path`, which is the path to the trained model that can be used for inference on the video frames.

---

### **`interpolate_ball_positions` Function**
```python
def interpolate_ball_positions(self, ball_positions):
    ball_positions = [x.get(1,[]) for x in ball_positions]
    df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
    df_ball_positions = df_ball_positions.interpolate()
    df_ball_positions = df_ball_positions.bfill()
    ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]
    return ball_positions
```
- **Purpose**: This function handles missing or incomplete ball position data by interpolating missing values between frames.
- **Process**:
  - **`ball_positions`**: This is a list of dictionaries, where each dictionary contains ball positions in a frame. The ball’s position is represented by a bounding box with coordinates `(x1, y1, x2, y2)` (top-left and bottom-right corners).
  - **Interpolation**: Missing values in the ball’s position (if any) are interpolated using the pandas `interpolate()` function.
  - **`bfill()`**: The `bfill()` function is used to fill any remaining missing values by backward filling.
  - **Output**: The ball positions are returned as a list of dictionaries with the interpolated coordinates.

---

### **`get_ball_shot_frames` Function**
```python
def get_ball_shot_frames(self, ball_positions):
    ball_positions = [x.get(1, []) for x in ball_positions]
    df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
    df_ball_positions['ball_hit'] = 0
    df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
    df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
    df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
    minimum_change_frames_for_hit = 25
    for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
        negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i + 1] < 0
        positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i + 1] > 0

        if negative_position_change or positive_position_change:
            change_count = 0
            for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0
                positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0

                if negative_position_change and negative_position_change_following_frame:
                    change_count += 1
                elif positive_position_change and positive_position_change_following_frame:
                    change_count += 1

            if change_count > minimum_change_frames_for_hit - 1:
                df_ball_positions['ball_hit'].iloc[i] = 1

    frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()
    return frame_nums_with_ball_hits
```
- **Purpose**: This function identifies frames where the ball is hit based on the change in its vertical position (`y` coordinate).
- **Process**:
  - **`df_ball_positions`**: Converts the ball positions into a pandas DataFrame.
  - **`mid_y`**: Calculates the vertical midpoint of the ball’s bounding box.
  - **Rolling Mean**: Computes a rolling mean of the vertical midpoint to smooth out noise in the vertical movement.
  - **`delta_y`**: The difference between consecutive rolling mean values (`mid_y`), which indicates movement.
  - **Ball Hit Detection**:
    - The function looks for significant changes in the `delta_y` (vertical movement) to detect ball hits (when the ball direction changes sharply).
    - It checks for a significant change in the ball's position over multiple frames (by counting consecutive frames with changes in direction).
  - **Output**: Returns the frame numbers where a ball hit is detected based on the position changes.

---

### **`detect_frames` Function**
```python
def detect_frames(self, frames, read_from_stub=False, stub_path=None):
    ball_detections = []

    if read_from_stub and stub_path is not None:
        with open(stub_path, 'rb') as f:
            ball_detections = pickle.load(f)
        return ball_detections

    for frame in frames:
        player_dict = self.detect_frame(frame)
        ball_detections.append(player_dict)

    if stub_path is not None:
        with open(stub_path, 'wb') as f:
            pickle.dump(ball_detections, f)

    return ball_detections
```
- **Purpose**: This function detects the ball in multiple frames.
- **Process**:
  - If `read_from_stub` is set to `True` and `stub_path` is provided, the function loads precomputed ball detection results from a file (using pickle) to save time.
  - If no stub is used, it processes each frame and calls `detect_frame()` to detect the ball.
  - The results are saved to a file using pickle, ensuring that they can be reused in the future to avoid recomputation.
  - **Output**: Returns a list of dictionaries with ball positions in each frame.

---

### **`detect_frame` Function**
```python
def detect_frame(self, frame):
    results = self.model.predict(frame, conf=0.15)[0]
    ball_dict = {}
    for box in results.boxes:
        result = box.xyxy.tolist()[0]
        ball_dict[1] = result
    return ball_dict
```
- **Purpose**: This function detects the ball in a single frame using the YOLO model.
- **Process**:
  - The `predict()` function of the YOLO model is called with the frame as input. It returns detection results, where each detected object is represented by a bounding box.
  - For each detected box (in this case, the ball), the function extracts the bounding box coordinates (`xyxy`) and stores them in a dictionary `ball_dict` with the key `1` (assuming only one object, the ball, is tracked).
  - **Output**: Returns a dictionary containing the bounding box coordinates of the detected ball.

---

### **`draw_bboxes` Function**
```python
def draw_bboxes(self, video_frames, player_detections):
    output_video_frames = []
    for frame, ball_dict in zip(video_frames, player_detections):
        for track_id, bbox in ball_dict.items():
            x1, y1, x2, y2 = bbox
            cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        output_video_frames.append(frame)
    return output_video_frames
```
- **Purpose**: This function draws bounding boxes around the detected ball in each video frame and annotates the frame with the ball's ID.
- **Process**:
  - For each frame and ball detection

, the function draws a rectangle around the detected ball using `cv2.rectangle()`.
  - It also labels the ball with its ID (assumed to be `1` in this case) using `cv2.putText()`.
  - **Output**: Returns the annotated video frames.

---

### **Summary**
The `BallTracker` class provides a robust framework for tracking and analyzing a ball's movement in a video using YOLO-based object detection. The key functionalities include detecting the ball in video frames, interpolating missing data, detecting ball hits based on vertical position changes, and annotating video frames with bounding boxes and labels. This can be useful for sports analytics, particularly in tracking the ball's movement in sports like tennis or soccer."""

from ultralytics import YOLO 
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_ball_shot_frames(self,ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_positions['ball_hit'].iloc[i] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    def detect_frame(self,frame):
        results = self.model.predict(frame,conf=0.15)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames


    