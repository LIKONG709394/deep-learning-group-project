# 🐍 AI Snake Kid

A classic Snake game that you can control with hand gestures using your webcam and a TensorFlow Lite model trained with Google's Teachable Machine.

**✨ NEW: Web-based UI with model upload feature!** Upload your own trained models directly in the browser.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.0+-green.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-red.svg)

## 🎮 Features

- 🎯 Classic Snake game built with Pygame
- 🤖 Real-time hand gesture recognition using TensorFlow Lite
- 📹 Webcam-based controls (up, down, left, right)
- ⌨️ Keyboard controls (arrow keys or WASD) as backup
- 🌐 **Modern web interface** with live camera feed and predictions
- 📤 **Upload your own models** - Train on Teachable Machine and upload instantly
- 🎨 Professional navy blue & white UI design
- 🔧 Modular design with separated control logic

## 📁 Project Structure

```
edupython/
├── app.py                   # Flask web server
├── game.py                  # Snake game logic (Pygame)
├── controller_tflite.py     # TFLite model inference & control
├── controls.py              # Shared control module
├── test_model.py            # Model testing utility
├── list_cameras.py          # Camera detection utility
├── templates/
│   └── index.html           # Web UI template
├── static/
│   └── game.js              # Frontend game logic
├── uploads/                 # User-uploaded models (gitignored)
├── model_unquant.tflite     # Default TFLite model
├── labels.txt               # Class labels (up, left, right, down)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Webcam (built-in or external)
- macOS, Linux, or Windows

### Installation

1. **Clone or download this repository**
   ```bash
   cd /path/to/edupython
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train your model** (if you haven't already)
   - Go to [Teachable Machine](https://teachablemachine.withgoogle.com/)
   - Create an Image Project with 4 classes:
     - `0 up` - Hand gesture for moving up
     - `1 left` - Hand gesture for moving left
     - `2 right` - Hand gesture for moving right
     - `3 bottom` - Hand gesture for moving down
   - Train with **50-100 images per class** in different lighting conditions
   - Export as **TensorFlow Lite** (Floating point)
   - Place `model_unquant.tflite` and `labels.txt` in project root

## 🎯 Usage

### 🌟 Option 1: Web Interface (Recommended)

The easiest way to play! Everything in one browser window:

```bash
source venv/bin/activate
python3 app.py
```

Then open your browser to: **http://127.0.0.1:8080**

**Features:**
- 🎮 Game on the left, camera feed on the right
- 📊 Live prediction visualization
- 📤 Upload your own trained models
- 🎨 Beautiful navy blue & white UI
- 🔄 No need to restart - models reload instantly

**How to upload your model:**
1. Train your model on [Teachable Machine](https://teachablemachine.withgoogle.com/)
2. Download the TFLite model and labels.txt
3. Click "Upload Your Model" section on the web page
4. Select both files and click "Upload and Load Model"
5. Wait a few seconds for the model to reload
6. Start playing!

### Option 2: Keyboard-Only Mode

Just play the game with your keyboard:

```bash
python3 game.py
```

- Use **arrow keys** or **WASD** to control the snake
- Press **R** to restart

### Option 3: Command Line Mode

Run the game and webcam controller in two separate terminals:

**Terminal 1 - Start the game:**
```bash
source venv/bin/activate
python3 game.py
```

**Terminal 2 - Start webcam controller (NOTE: will force the notebook built-in camera by default):**
```bash
source venv/bin/activate
python3 controller_tflite.py --camera
```

By default the controller will only open the notebook's built-in camera (camera ID 0) to avoid accidentally using an external device (e.g. phone camera). If you really want to allow external or phone cameras you must opt-in with `--allow-external`. You can also enable automatic scanning of available devices with `--scan` (only when `--allow-external` is used):

```bash
# Allow external camera and auto-scan to find one
python3 controller_tflite.py --camera --allow-external --scan

# Explicitly use camera ID 1 (must also allow external if it's not the built-in)
python3 controller_tflite.py --camera --camera-id 1 --allow-external
```

### Advanced Options

**Use a specific camera** (if you have multiple webcams):
```bash
# List available cameras
python3 list_cameras.py

# Use camera ID 0 (built-in, recommended)
python3 controller_tflite.py --camera --camera-id 0

# To use an external camera (e.g. phone), you MUST opt in
python3 controller_tflite.py --camera --camera-id 1 --allow-external
```

**Adjust confidence threshold** (default 0.5):
```bash
python3 controller_tflite.py --camera --threshold 0.7
```

**Test with a single image:**
```bash
python3 controller_tflite.py --image path/to/your/image.jpg
```

**Test model in real-time** (shows predictions on screen):
```bash
python3 test_model.py
```

## 🔧 Troubleshooting

### Model Not Predicting Correctly

**Symptoms:** Predictions don't change or are always the same class

**Solution:**
1. Retrain your model with more diverse samples (50-100 per class)
2. Ensure each gesture is visually distinct
3. Test in Teachable Machine preview before exporting
4. Use consistent lighting during training and inference

### Camera Permission Issues

**macOS:** System Preferences → Security & Privacy → Camera → Allow Terminal/Python

**Linux:** Ensure your user is in the `video` group:
```bash
sudo usermod -a -G video $USER
```

### Virtual Environment Issues

If `pip` is broken, recreate the environment:
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Wrong Camera Being Used

1. Run `python3 list_cameras.py` to see all available cameras
2. Use `--camera-id` flag to specify the correct one
3. Built-in MacBook camera is usually ID 0
4. iPhone Continuity Camera is usually ID 1

## 📋 Label Format

Your `labels.txt` should contain 4 lines with class names:

```
0 up
1 left
2 right
3 bottom
```

The controller will map any label containing these keywords:
- `up` → Move snake up
- `left` → Move snake left
- `right` → Move snake right
- `bottom` or `down` → Move snake down

## 🎨 Customization

### Game Settings (game.py)

```python
WINDOW_SIZE = 400      # Game window size (pixels)
TILE_COUNT = 20        # Grid size (20x20)
MOVE_INTERVAL = 120    # Snake speed (milliseconds)
```

### Controller Settings (controller_tflite.py)

```python
IMG_SIZE = (224, 224)         # Model input size
PRED_INTERVAL = 0.12          # Prediction frequency (seconds)
CONF_THRESHOLD = 0.5          # Minimum confidence to accept prediction
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Google Teachable Machine](https://teachablemachine.withgoogle.com/) - Easy ML model training
- [TensorFlow Lite](https://www.tensorflow.org/lite) - Efficient model inference
- [Pygame](https://www.pygame.org/) - Game development framework

## 📞 Support

If you encounter any issues:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Run `python3 test_model.py` to verify model is working
3. Run `python3 list_cameras.py` to check camera setup

---

**Enjoy playing Snake with AI! 🎮🐍🤖**
