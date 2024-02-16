# Smarthome Gesture Control ML

## Project Setup

To contribute or run the Smarthome Gesture Control ML project, follow the setup instructions below:

### Prerequisites
- Any Python-based editor (e.g., VSCode, PyCharm) is recommended as the Integrated Development Environment (IDE).
- Python 3.8 is **mandatory** for this project.
- Install the following Python modules:
  - TensorFlow
  - Python 3.6.9
  - OpenCV for Python
  - Keras

## Run Locally

To run the application locally, use one of the following methods:

1. **Test Videos:**
   - Place your test videos in the `test` folder.

2. **Training Videos:**
   - Training videos can be found in the `traindata` folder.

3. **Command to Execute:**
   - Run the program with the following command:
     ```bash
     python main.py
     ```

4. **Frame Extraction:**
   - Frames will be extracted and stored in a new folder named `frames` within the respective video folder.

5. **Gesture Output Labels:**
   - The output labels for gestures will be available in `results.csv` with the header in the column named `Output Label`.

