# NFL Broadcast Commercial Detector

A transfer learning computer vision system that automatically detects transitions between NFL game footage and commercials in real-time, with automatic audio muting during commercial breaks.

Note: Parts of the code (including the remainder of the README) was AI-generated.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.8+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Features

- **🎥 Chrome Tab Capture**: Direct browser tab capture using Selenium WebDriver
- **🤖 Real-time AI Detection**: ResNet-50 model classifies content as "game" or "commercial"
- **🔇 Automatic Muting**: Mutes Chrome during commercials (Windows)
- **⚙️ Configurable**: Adjustable confidence thresholds, cooldown periods, and smoothing
- **📸 Transition Logging**: Saves screenshots and logs all detected transitions
- **🎯 Smart Filtering**: Uses confidence thresholds to reduce false positives
- **💾 Persistent Profile**: Sign in once - Chrome remembers your logins

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- **Windows OS** (currently required for auto-mute feature)
- **Google Chrome browser** (currently required for tab capture)
- NVIDIA GPU (optional, recommended for faster inference)

> **Note**: This tool currently requires **Windows and Chrome**, but can be easily adapted for other platforms and browsers. The core detection logic is platform-agnostic - only the audio control (Windows-specific) and browser automation (Chrome-specific) would need to be modified for cross-platform support.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/scottn12/NFL_Detector.git
   cd NFL_Detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   This automatically installs ChromeDriver via `webdriver-manager` - no manual setup needed!

3. **You're ready to go!**
   - Use the included `nfl_classifier_best.pth` model, or
   - Train your own model using `train_nfl_classifier.py`

### Basic Usage

1. **Run the detector**
   ```bash
   python detector.py
   ```

2. **The system will:**
   - Automatically start Chrome in debug mode (if not already running)
   - Use a persistent profile (sign in once, stays logged in)
   - Prompt you to navigate to your NFL stream
   
3. **Navigate to your stream**
   - Open your NFL stream in the Chrome window that appears
   - Start the video playing
   - Come back and press ENTER

4. **Automatic detection begins!**
   - Monitors the Chrome tab for game/commercial transitions
   - Automatically mutes Chrome during commercials
   - Unmutes when game resumes
   - Press `Ctrl+C` to stop (audio automatically restored)

### Example Session

```bash
$ python detector.py
======================================================================
NFL DETECTOR - CHROME TAB CAPTURE
======================================================================
Chrome will start automatically in debug mode.
After Chrome opens, navigate to your NFL stream.
----------------------------------------------------------------------

Loading model on cpu...
   Chrome not running in debug mode. Starting Chrome...
   Waiting for Chrome to start...
   ✓ Chrome is ready! (took 1s)
   Connecting to Chrome on port 9222...
   Initializing WebDriver (this may take a moment)...
✓ Chrome WebDriver initialized (tab capture enabled)
✓ Audio control will be initialized when stream starts playing
✓ Model loaded successfully!

======================================================================
CHROME IS READY!
======================================================================

📺 NEXT STEPS:
1. A Chrome window should be open
2. Navigate to your NFL stream in that Chrome window
3. Start the video playing
4. You can use fullscreen for best results
5. Come back here and press ENTER to start detection

💡 TIP: The detector captures directly from the Chrome tab.
   Keep the Chrome window/tab open during detection!
======================================================================

Press ENTER when your stream is playing and ready...
```

## 📊 Configuration

### Key Parameters

Edit settings in `config.py` to tune performance:

```python
DETECTION_CONFIG = {
    'threshold': 0.6,              # Classification threshold (0.5-0.7)
    'cooldown_seconds': 20,        # Min time between transitions (10-30s)
    'check_interval': 1.0,         # Check frequency in seconds
    'smoothing_window': 5,         # Number of predictions to average (3-7)
    'auto_mute': True,             # Enable automatic muting
    'min_confidence': 0.90,        # Minimum confidence for transitions (0.8-0.95)
    'control_media': False,        # Play Spotify/music during commercials
}
```

### Reducing False Positives

If you're getting too many false positives:

1. **Increase `min_confidence`** to 0.95
2. **Increase `threshold`** to 0.65-0.7
3. **Increase `cooldown_seconds`** to 30
4. **Increase `smoothing_window`** to 7

### Media Control (Optional)

Set `control_media: True` in config to automatically:
- ▶️ Play system media (Spotify, etc.) during commercials
- ⏸️ Pause media when game resumes

## 🏗️ Project Structure

```
NFL_Detector/
├── detector.py              # Main detection script
├── config.py                # Configuration settings
├── train_nfl_classifier.py  # Model training script
├── nfl_classifier_best.pth  # Pre-trained model
├── requirements.txt         # Python dependencies
├── data/                    # Training data
│   ├── train/
│   │   ├── game/           # Game footage images
│   │   └── commercial/     # Commercial images
│   ├── val/                # Validation data
│   └── test/               # Test data
├── screenshot_extractor_tool/  # Video processing utility
└── transitions/            # Saved transition screenshots
```

## 🧠 Model Training

### Preparing Training Data

1. **Collect Screenshots**
   ```bash
   cd screenshot_extractor_tool
   python video_screenshot_extractor.py your_video.mp4
   ```

2. **Organize Data**
   ```
   data/
   ├── train/
   │   ├── game/      # Put game screenshots here
   │   └── commercial/  # Put commercial screenshots here
   ├── val/
   │   ├── game/
   │   └── commercial/
   └── test/
       ├── game/
       └── commercial/
   ```

3. **Train the Model**
   ```bash
   python train_nfl_classifier.py
   ```

### Model Architecture

- **Base**: ResNet-50 (pre-trained on ImageNet)
- **Custom Head**: 
  - Linear(2048 → 256)
  - ReLU + Dropout(0.3)
  - Linear(256 → 1)
  - Sigmoid activation
- **Input**: 224×224 RGB images
- **Output**: Probability (0 = commercial, 1 = game)

## 📈 Performance Monitoring

### Real-time Status

The detector shows live statistics:

```
[0042] Current: GAME        | Duration: 02:34 | Confidence: 94.2% | Transitions: 3 | Audio: 🔊 UNMUTED
```

### Transition Logs

When transitions occur, detailed logs are shown:

```
======================================================================
🔄 TRANSITION DETECTED at 2025-10-16 14:23:45
   GAME → COMMERCIAL
   Previous state duration: 156.3 seconds
   Confidence: 92.1%
   Total transitions: 4
   🔇 MUTED
======================================================================
```

### Saved Screenshots

Transition screenshots are automatically saved in `transitions/`:
- `20251016_142345_game_to_commercial.jpg`
- `20251016_142620_commercial_to_game.jpg`

## 🛠️ Troubleshooting

### Chrome Issues

**"Chrome failed to start in debug mode"**
- Close all Chrome windows manually: `taskkill /F /IM chrome.exe`
- Run `python detector.py` again (Chrome starts automatically)

**"Chrome tab capture failed"**
- Make sure the video is playing in Chrome
- Navigate to your stream and press play
- The detector needs an active Chrome tab to capture

**"Could not initialize audio control"**
- Audio control initializes when Chrome starts playing audio
- Make sure your stream is playing with sound
- Check Windows audio permissions

### Detection Issues

**High false positive rate**
- Increase `min_confidence` to 0.95 in `config.py`
- Increase `threshold` to 0.65-0.7
- Increase `cooldown_seconds` to 30

**Model not detecting transitions**
- Lower `min_confidence` to 0.80 in `config.py`
- Check if `nfl_classifier_best.pth` exists
- Verify video is playing in Chrome

### Performance Tips

- **GPU Acceleration**: Install CUDA-compatible PyTorch for faster inference
- **Check Interval**: Increase to 2-3 seconds in config if CPU usage is high
- **Fullscreen**: Use fullscreen video for best detection accuracy

## 📋 Dependencies

### Core Dependencies
- **PyTorch** (≥1.8.0): Deep learning framework
- **torchvision** (≥0.9.0): Computer vision utilities
- **Pillow** (≥8.0.0): Image processing
- **Selenium** (≥4.0.0): Browser automation and tab capture
- **webdriver-manager** (≥4.0.0): Automatic ChromeDriver management
- **pycaw**: Windows audio control
- **comtypes**: Windows COM interface

### Training Dependencies (optional)
- **scikit-learn** (≥0.24.0): Model evaluation
- **numpy** (≥1.20.0): Numerical computing

All dependencies are installed automatically via `pip install -r requirements.txt`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ResNet-50**: Pre-trained model from PyTorch Model Zoo
- **pycaw**: Windows audio control library
- **mss**: Cross-platform screen capture

## 📞 Support

If you encounter issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing [Issues](https://github.com/scottn12/NFL_Detector/issues)
3. Create a new issue with detailed information

---

**Note**: This tool is designed for personal use to automatically mute commercials during NFL broadcasts. Ensure compliance with your local laws and streaming service terms of use.
