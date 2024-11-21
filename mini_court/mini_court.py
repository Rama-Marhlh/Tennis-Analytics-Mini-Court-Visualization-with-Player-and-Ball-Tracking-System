"""
This code is a Python implementation of a **MiniCourt** class using the OpenCV and NumPy libraries. This class visually represents a scaled-down version of a sports court (e.g., a tennis or badminton court) overlaid on video frames. It also maps real-world positions (e.g., players, balls) to this mini court, enabling the tracking of their locations in a compact visual representation.

Here's a detailed explanation of the code:

---

### **Imports**

- **`cv2` and `numpy`**: Used for image processing and numerical computations.
- **`sys.path.append('../')`**: Adds the parent directory to the import path, enabling custom utility functions and constants from other files to be used.

---

### **Utility Functions**
The code relies on external utility functions (`utils`) to perform specific tasks like:
- Converting between meters and pixels.
- Measuring distances (e.g., Euclidean distance).
- Identifying keypoints.
- Bounding box-related computations.

---

### **Class: MiniCourt**

This class is designed to create, draw, and manage a miniaturized version of a court. The court is rendered on video frames for visualization and tracking of players and the ball.

#### **Initialization (`__init__`)**
```python
def __init__(self, frame):
```
- Initializes the MiniCourt object with a given video frame.
- Sets default dimensions and positions for the mini court and background rectangle.
- Calls helper methods to configure:
  1. **Canvas background** (`set_canvas_background_box_position`): The area where the court will be drawn.
  2. **Court position** (`set_mini_court_position`): Defines the position of the court relative to the canvas.
  3. **Key points** (`set_court_drawing_key_points`): Computes the positions of essential court lines and boundaries.
  4. **Court lines** (`set_court_lines`): Defines the connections between key points for drawing the court.

---

#### **Key Methods**

1. **`convert_meters_to_pixels`**
   Converts real-world distances (in meters) to pixel values based on the court's dimensions.
   ```python
   def convert_meters_to_pixels(self, meters):
       return convert_meters_to_pixel_distance(meters,
                                               constants.DOUBLE_LINE_WIDTH,
                                               self.court_drawing_width)
   ```

2. **`set_court_drawing_key_points`**
   Computes key points (28 in total) for drawing the mini court using constants from a configuration file.
   - **Key points**: Specific locations representing court boundaries, no-man's land, service lines, etc.
   - Uses pixel-to-meter conversion to ensure the court's proportions remain accurate.

3. **`set_court_lines`**
   Defines pairs of points that form lines connecting the key points, which collectively represent the court's structure.
   ```python
   self.lines = [
       (0, 2),  # Top boundary
       (4, 5),  # Half-court line
       ...
   ]
   ```

4. **`set_mini_court_position`**
   Positions the court within the background rectangle by adding padding to the start and end coordinates.

5. **`set_canvas_background_box_position`**
   Defines the background rectangle's position and dimensions based on the input frame size.

6. **`draw_court`**
   - Draws the mini court on a video frame using:
     - Circles to mark key points.
     - Lines to connect key points.
     - A horizontal blue line to represent the net.
   ```python
   cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Draw key points
   cv2.line(frame, start_point, end_point, (0, 0, 0), 2)  # Draw court lines
   ```

7. **`draw_background_rectangle`**
   Creates a semi-transparent background rectangle on the frame where the mini court will be displayed.

8. **`draw_mini_court`**
   Combines the background rectangle and court drawing, applying these to all frames in a sequence.

9. **`get_mini_court_coordinates`**
   Maps an object's position (e.g., a player or ball) from the full court to the mini court's coordinate system:
   - **Steps**:
     1. Calculate the object's distance from a key point (in pixels).
     2. Convert the pixel distance to meters.
     3. Map this distance back to the mini court.
   - **Returns**: The object's position in mini court coordinates.

10. **`convert_bounding_boxes_to_mini_court_coordinates`**
    Converts bounding box coordinates for players and the ball in the original court to their corresponding positions on the mini court:
    - Loops through all frames.
    - Calculates the center of bounding boxes and maps them to the mini court.
    - Identifies the player closest to the ball for tracking.

11. **`draw_points_on_mini_court`**
    Draws player and ball positions (from the mini court) onto frames as small colored circles.

---

### **Constants Used**
The code uses constants like:
- `DOUBLE_LINE_WIDTH`: Width of lines on the court.
- `HALF_COURT_LINE_HEIGHT`: Height of the half-court area.
- `NO_MANS_LAND_HEIGHT`: Distance of the no-man's land.
- Player heights (`PLAYER_1_HEIGHT_METERS`, etc.).

These ensure proportions and scaling remain consistent.

---

### **Usage Flow**
1. **Initialize**:
   Create a `MiniCourt` object with a frame.
2. **Draw Court**:
   Use `draw_mini_court` to overlay the mini court on each frame.
3. **Map Positions**:
   - Use `get_mini_court_coordinates` or `convert_bounding_boxes_to_mini_court_coordinates` to map real-world positions to the mini court.
   - Draw these positions using `draw_points_on_mini_court`.

---

### **Key Features**
- **Dynamic Resizing**: The court adjusts its size and position based on the input frame.
- **Real-World Mapping**: Converts real-world distances and positions into scaled coordinates on the mini court.
- **Visual Overlay**: Provides a clear, intuitive representation of players and ball positions for tracking and analysis.
"""
import cv2
import numpy as np
import sys
sys.path.append('../')
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance
)

class MiniCourt():
    def __init__(self,frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court=20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()


    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width
                                            )

    def set_court_drawing_key_points(self):
        drawing_key_points = [0]*28

        # point 0 
        drawing_key_points[0] , drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_key_points[2] , drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT*2)
        # point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5] 
        # #point 4
        drawing_key_points[8] = drawing_key_points[0] +  self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 
        # #point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5] 
        # #point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 
        # #point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7] 
        # #point 8
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17] 
        # #point 10
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 11
        drawing_key_points[22] = drawing_key_points[20] +  self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21] 
        # # #point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 
        # # #point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21] 

        self.drawing_key_points=drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),
            (8,9),
            (10,11),
            (10,11),
            (2,3)
        ]

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self,frame):
        frame= frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_court(self,frame):
        for i in range(0, len(self.drawing_key_points),2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x,y),5, (0,0,255),-1)

        # draw Lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    def draw_background_rectangle(self,frame):
        shapes = np.zeros_like(frame,np.uint8)
        # Draw the rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha=0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out

    def draw_mini_court(self,frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    def get_start_point_of_mini_court(self):
        return (self.court_start_x,self.court_start_y)
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    def get_mini_court_coordinates(self,
                                   object_position,
                                   closest_key_point, 
                                   closest_key_point_index, 
                                   player_height_in_pixels,
                                   player_height_in_meters
                                   ):
        
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_key_point)

        # Conver pixel distance to meters
        distance_from_keypoint_x_meters = convert_pixel_distance_to_meters(distance_from_keypoint_x_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels
                                                                           )
        distance_from_keypoint_y_meters = convert_pixel_distance_to_meters(distance_from_keypoint_y_pixels,
                                                                                player_height_in_meters,
                                                                                player_height_in_pixels
                                                                          )
        
        # Convert to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_y_meters)
        closest_mini_coourt_keypoint = ( self.drawing_key_points[closest_key_point_index*2],
                                        self.drawing_key_points[closest_key_point_index*2+1]
                                        )
        
        mini_court_player_position = (closest_mini_coourt_keypoint[0]+mini_court_x_distance_pixels,
                                      closest_mini_coourt_keypoint[1]+mini_court_y_distance_pixels
                                        )

        return  mini_court_player_position

    def convert_bounding_boxes_to_mini_court_coordinates(self,player_boxes, ball_boxes, original_court_key_points ):
        player_heights = {
            1: constants.PLAYER_1_HEIGHT_METERS,
            2: constants.PLAYER_2_HEIGHT_METERS
        }

        output_player_boxes= []
        output_ball_boxes= []

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num][1]
            ball_position = get_center_of_bbox(ball_box)
            closest_player_id_to_ball = min(player_bbox.keys(), key=lambda x: measure_distance(ball_position, get_center_of_bbox(player_bbox[x])))

            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                # Get The closest keypoint in pixels
                closest_key_point_index = get_closest_keypoint_index(foot_position,original_court_key_points, [0,2,12,13])
                closest_key_point = (original_court_key_points[closest_key_point_index*2], 
                                     original_court_key_points[closest_key_point_index*2+1])

                # Get Player height in pixels
                frame_index_min = max(0, frame_num-20)
                frame_index_max = min(len(player_boxes), frame_num+50)
                bboxes_heights_in_pixels = [get_height_of_bbox(player_boxes[i][player_id]) for i in range (frame_index_min,frame_index_max)]
                max_player_height_in_pixels = max(bboxes_heights_in_pixels)

                mini_court_player_position = self.get_mini_court_coordinates(foot_position,
                                                                            closest_key_point, 
                                                                            closest_key_point_index, 
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id]
                                                                            )
                
                output_player_bboxes_dict[player_id] = mini_court_player_position

                if closest_player_id_to_ball == player_id:
                    # Get The closest keypoint in pixels
                    closest_key_point_index = get_closest_keypoint_index(ball_position,original_court_key_points, [0,2,12,13])
                    closest_key_point = (original_court_key_points[closest_key_point_index*2], 
                                        original_court_key_points[closest_key_point_index*2+1])
                    
                    mini_court_player_position = self.get_mini_court_coordinates(ball_position,
                                                                            closest_key_point, 
                                                                            closest_key_point_index, 
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id]
                                                                            )
                    output_ball_boxes.append({1:mini_court_player_position})
            output_player_boxes.append(output_player_bboxes_dict)

        return output_player_boxes , output_ball_boxes
    
    def draw_points_on_mini_court(self,frames,postions, color=(0,255,0)):
        for frame_num, frame in enumerate(frames):
            for _, position in postions[frame_num].items():
                x,y = position
                x= int(x)
                y= int(y)
                cv2.circle(frame, (x,y), 5, color, -1)
        return frames

