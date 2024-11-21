"""This code defines a class `CourtLineDetector` that utilizes a pre-trained ResNet-50 model to detect keypoints on an image ( to detect court lines on a tennis court). The keypoints represent important locations on the court, such as the positions of the lines or boundaries. The class performs two main functions: **predicting the keypoints** for a given image and **drawing these keypoints** on the image or video frames. Below is a detailed breakdown of the code:

### 1. **Class Initialization (`__init__`)**
The constructor (`__init__`) is responsible for setting up the model and the necessary image transformations:
```python
def __init__(self, model_path):
    self.model = models.resnet50(pretrained=True)  # Load a pre-trained ResNet-50 model.
    self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)  # Modify the final layer of the model to output 14 keypoints (each with x and y coordinates).
    self.model.load_state_dict(torch.load(model_path, map_location='cpu'))  # Load the custom-trained weights from the specified model path.
    
    # Define a series of transformations to apply to the input image.
    self.transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert the input image (NumPy array) to a PIL image.
        transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels to match the input size for ResNet-50.
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image with standard values for ImageNet (pre-trained weights).
    ])
```

#### Explanation:
- **ResNet-50 Model**: ResNet-50 is a pre-trained deep learning model, often used for image classification tasks. The pre-trained weights are loaded from ImageNet.
- **Final Layer Modification**: The last fully connected (fc) layer of the ResNet-50 is modified so that it outputs `14*2` values, which are the `(x, y)` coordinates of 14 keypoints (hence, `14*2` values for `x` and `y` pairs).
- **Image Transformations**: Before feeding the image into the model, the image is converted into a format suitable for the model (e.g., resizing, normalization).

### 2. **Prediction Method (`predict`)**
The `predict` function processes an image and predicts the court keypoints (coordinates) from it:
```python
def predict(self, image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert the input image from BGR (OpenCV format) to RGB.
    image_tensor = self.transform(image_rgb).unsqueeze(0)  # Apply transformations and add a batch dimension (unsqueeze).
    
    with torch.no_grad():  # Disable gradient computation for inference (saves memory and computation).
        outputs = self.model(image_tensor)  # Pass the image tensor through the model.
    
    keypoints = outputs.squeeze().cpu().numpy()  # Remove the batch dimension, move the tensor to CPU, and convert it to a NumPy array.
    
    original_h, original_w = image.shape[:2]  # Get the original dimensions of the image (height and width).
    
    # Rescale the predicted keypoints from the 224x224 resolution to the original image resolution.
    keypoints[::2] *= original_w / 224.0  # Rescale x-coordinates.
    keypoints[1::2] *= original_h / 224.0  # Rescale y-coordinates.

    return keypoints
```

#### Explanation:
- **BGR to RGB**: OpenCV loads images in BGR format, but most deep learning models (like ResNet) expect RGB format, so the image is converted.
- **Image Transformations**: The image is transformed using the previously defined pipeline (resize, normalization).
- **Keypoint Prediction**: The transformed image is passed through the modified ResNet-50 model, which outputs a tensor of predicted keypoints.
- **Rescaling**: The model's input is resized to 224x224 pixels, so the predicted keypoints need to be rescaled back to the original image dimensions (height and width).

### 3. **Drawing Keypoints on Image (`draw_keypoints`)**
The `draw_keypoints` function takes an image and the predicted keypoints and draws the keypoints on the image:
```python
def draw_keypoints(self, image, keypoints):
    # Plot keypoints on the image
    for i in range(0, len(keypoints), 2):  # Iterate through the keypoints (x, y) pairs.
        x = int(keypoints[i])  # Get the x-coordinate.
        y = int(keypoints[i+1])  # Get the y-coordinate.
        
        cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Add a label with the keypoint index above the point.
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Draw a red circle at the keypoint position.
    
    return image
```

#### Explanation:
- **Loop Through Keypoints**: The keypoints are stored as a flattened list, alternating between x and y coordinates. The loop extracts each pair and draws a circle and a label on the image at the corresponding `(x, y)` location.
- **Circle and Label**: For each keypoint, a small red circle is drawn, and the index of the keypoint is displayed above it.

### 4. **Drawing Keypoints on Video (`draw_keypoints_on_video`)**
This method processes a list of video frames and draws the detected keypoints on each frame:
```python
def draw_keypoints_on_video(self, video_frames, keypoints):
    output_video_frames = []  # List to hold the frames with keypoints drawn.
    
    for frame in video_frames:
        frame = self.draw_keypoints(frame, keypoints)  # Draw keypoints on the frame.
        output_video_frames.append(frame)  # Append the frame with keypoints to the output list.
    
    return output_video_frames
```

#### Explanation:
- **Multiple Frames**: This method iterates over a list of video frames, applying the `draw_keypoints` function to each frame.
- **Return Processed Frames**: It returns the list of frames, each with the keypoints drawn.

### Key Components:
- **ResNet-50 Model**: A pre-trained model used for feature extraction and custom keypoint detection.
- **Image Transformations**: Pre-processing steps to make the input suitable for the model.
- **Keypoint Rescaling**: Adjusting predicted keypoints to match the original image dimensions.
- **Drawing Keypoints**: Visualizing the detected keypoints on the image or video frames.

### Usage:
- **Input**: The input is either a single image or a sequence of video frames.
- **Output**: The output is either the keypoints (for a single image) or a sequence of frames with keypoints drawn on them.

This class is primarily used for detecting and visualizing important points (keypoints) on an image or video, typically for sports analysis or court line detection."""
import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) 
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):

    
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        return keypoints

    def draw_keypoints(self, image, keypoints):
        # Plot keypoints on the image
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames