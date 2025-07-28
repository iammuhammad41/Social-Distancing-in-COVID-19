# Social Distancing Detection in COVID-19 using Deep Learning (YOLO)

## Project Overview

This project utilizes deep learning methods, particularly **YOLO (You Only Look Once)**, to detect social distancing violations in public spaces. The model processes video footage of pedestrians, tracks individuals, and evaluates their proximity to ensure compliance with social distancing guidelines during the COVID-19 pandemic. The software analyzes the input video, generates a bird-eye view of the scene, and displays real-time results, identifying social distancing violations by measuring the distance between individuals.

### Key Features:

1. **Pedestrian Detection**: Using the YOLOv3 model to detect pedestrians.
2. **Centroid Tracker**: Tracks the individuals in the video frame by frame.
3. **Social Distancing Evaluation**: Calculates the distance between pedestrians and highlights social distancing violations.
4. **Bird's Eye View**: Visualizes the scenario from a top-down perspective.
5. **Real-time Processing**: The system processes the video in real-time to detect and visualize violations.

### Requirements

* Python 3.x
* OpenCV
* NumPy
* imutils
* albumentations
* PyTorch (for deep learning models)
* matplotlib

You can install the required Python libraries using the following:

```bash
pip install opencv-python-headless imutils albumentations matplotlib numpy
```

### Project File Structure

```
Social-Distancing-Detection/
│
├── video_analysis.py         # Main script for detecting social distancing violations in video
├── utils.py                 # Utility functions for transformations and distance calculation
├── README.md                # Project README file
└── requirements.txt         # Python dependencies
```

### 1. **`video_analysis.py`**

This file handles the video processing, pedestrian detection, tracking, and social distancing calculations. It takes a video file as input, processes each frame, and outputs the results in a video file and a bird-eye view.

### 2. **`utils.py`**

This file contains utility functions for the project:

* `get_transformed_points()`: Transforms coordinates using a perspective matrix.
* `cal_dis()`: Calculates the distance between two points.
* `get_distances()`: Calculates distances for multiple pairs of detected pedestrians.
* `get_count()`: Counts the number of violations based on the distances between individuals.
* `bird_eye_view()`: Generates the bird-eye view visualization.
* `social_distancing_view()`: Overlays the social distancing information on the video frame.

### 3. **`CentroidTracker` Class**

A tracking algorithm for monitoring individuals in each frame. It registers and updates object IDs to track pedestrians and ensures accurate counting.

### 4. **Usage Instructions**

To run the project:

1. Download the YOLOv3 pre-trained weights and configuration file.

   * **YOLOv3 weights**: [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
   * **YOLOv3 config**: [yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)

2. Place the video (e.g., `Pedestrian 480p.mp4`) in the project directory.

3. Run the `video_analysis.py` script:

```bash
python video_analysis.py
```

The script will output a video showing detected pedestrians, their distances, and any violations of social distancing.

### 5. **Model Architecture**

The model used for pedestrian detection is **YOLOv3**, a convolutional neural network designed for object detection tasks. The architecture works by dividing the image into a grid and predicting bounding boxes for each cell. YOLOv3 achieves great accuracy and speed, making it suitable for real-time video processing.

The **CentroidTracker** class is responsible for tracking individuals over multiple frames. It associates detected objects (pedestrians) from one frame to the next based on their centroids (center points).

#### Spatial Attention:

The model doesn't directly implement spatial attention mechanisms; however, the bird-eye view projection can be considered a form of attention since it focuses on the relationships between detected individuals, highlighting violations of social distancing in the scene.

### 6. **Results**

* The system outputs two types of results:

  * **Bird-eye View**: A top-down perspective of the scene with connections between pedestrians based on their distance.
  * **Social Distancing Violations**: Individuals who are closer than a threshold distance (e.g., 150 pixels) are marked as violating social distancing.

### Example Output:

* The system will print real-time logs such as:

  ```
  Jumlah Orang Terdeteksi : 10 Orang
  Jumlah Pelanggaran Social Distancing : 2 Orang
  ```

* Additionally, it will save a video showing the bird-eye view and social distancing violations.

### 7. **Future Improvements**

* Integration with more complex tracking algorithms.
* Automatic adjustment of the minimum distance threshold based on environmental factors (crowd density, location, etc.).
* Enhanced visualization tools for better presentation of results.

### Reference:

This project leverages the YOLOv3 model for object detection, which can be referenced at [YOLOv3 GitHub](https://github.com/pjreddie/darknet) for further reading.

---

### **requirements.txt**

```
opencv-python-headless
imutils
albumentations
matplotlib
numpy
torch
```
