# Sports Broadcast Commercial Detector

A transfer learning computer vision system that automatically detects transitions between sports game footage and commercials in real-time, with automatic audio muting during commercial breaks.

Note: Parts of the code (including most of the README) was AI-generated.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.8+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Features

- **🏅 Multi-Sport Support**: Works with any sport - NFL, NBA, MLB, NHL, and more!
- **🎥 Chrome Tab Capture**: Direct browser tab capture using Selenium WebDriver
- **🤖 Real-time AI Detection**: ResNet-50 model classifies content as "game" or "commercial"
- **🔇 Automatic Muting**: Mutes Chrome during commercials (Windows only)
- **🎵 Media Control**: Automatically play/pause system media (Spotify, etc.) during transitions
- **⚙️ Configurable**: Adjustable confidence thresholds, cooldown periods, and smoothing
- **📸 Transition Logging**: Saves screenshots and logs all detected transitions (optional)
- **🎯 Smart Filtering**: Uses confidence thresholds and smoothing to reduce false positives
- **💾 Persistent Profile**: Sign in once - Chrome remembers your logins
- **🔄 Automatic Chrome Setup**: Automatically starts Chrome in debug mode for seamless capture

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
   git clone https://github.com/scottn12/Sports-Broadcast-Commercial-Detector.git
   cd Sports-Broadcast-Commercial-Detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   This automatically installs ChromeDriver via `webdriver-manager` - no manual setup needed!

3. **Download or train a model**
   
   > **Note**: Pre-trained models are **not included** in this repository due to their large file size (~92 MB each). You have two options:
   
   - **Option A**: Train your own model using `train_sports_classifier.py` (see [Model Training](#-model-training) section)
   - **Option B**: In the future, pre-trained models may be made available in the [Releases](https://github.com/scottn12/Sports-Broadcast-Commercial-Detector/releases) section. If available, download them and place them in the `models/` directory.
   
   Models should be placed in a `models/` directory with sport-specific names (e.g., `models/nfl_classifier_best.pth`, `models/nba_classifier_best.pth`).

4. **You're ready to go!**

> **⚠️ WARNING:** When you run the detector, it will **automatically close ALL Chrome instances** to start Chrome in debug mode. Make sure to save any work in Chrome before running the detector!

### Basic Usage

1. **Run the detector**
   ```bash
   # For NFL (default)
   python detector.py
   
   # For NBA
   python detector.py --sport nba
   
   # For MLB
   python detector.py --sport mlb
   
   # For NHL
   python detector.py --sport nhl
   ```

2. **Chrome starts automatically**
   - The detector automatically closes any existing Chrome instances
   - Starts Chrome in debug mode with a persistent profile
   - Uses a separate profile directory (sign in once, stays logged in)
   - Waits for Chrome to be ready
   
3. **Navigate to your stream**
   - In the Chrome window that appears, go to your sports stream
   - Start the video playing (fullscreen recommended)
   - Return to the terminal and press ENTER

4. **Automatic detection begins!**
   - Monitors the Chrome tab for game/commercial transitions
   - Automatically mutes Chrome during commercials
   - Unmutes when game resumes
   - Optionally plays/pauses system media (Spotify, etc.)
   - Press `Ctrl+C` to stop (audio and media automatically restored)

### Example Session

```bash
$ python detector.py --sport nfl
======================================================================
NFL DETECTOR - CHROME TAB CAPTURE
======================================================================
Chrome will start automatically in debug mode.
After Chrome opens, navigate to your NFL stream.
----------------------------------------------------------------------

Loading NFL model on cpu...
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
# Detection Parameters
DETECTION_CONFIG = {
    'threshold': 0.6,              # Classification threshold (0.5-0.7)
    'cooldown_seconds': 0,         # Min time between transitions (0-30s, 0 = disabled)
    'check_interval': 1.0,         # Check frequency in seconds
    'smoothing_window': 5,         # Number of predictions to average (3-7)
    'auto_mute': True,             # Enable automatic Chrome muting
    'min_confidence': 0.90,        # Minimum confidence for transitions (0.8-0.95)
    'control_media': True          # Play/pause system media during transitions
}

# Runtime Configuration  
RUNTIME_CONFIG = {
    'duration': None,              # Runtime limit in seconds (None = run forever)
    'save_transitions': False      # Save screenshots of transitions
}
```

### Reducing False Positives

If you're getting too many false positives:

1. **Increase `min_confidence`** to 0.95 (from 0.90)
2. **Increase `threshold`** to 0.65-0.7 (from 0.6)
3. **Enable `cooldown_seconds`** to 20-30 (default is 0 = disabled)
4. **Increase `smoothing_window`** to 7 (from 5)

### Media Control

Enable or disable media control in `config.py`:

```python
'control_media': True    # Toggle system media during transitions
```

When enabled, the detector will:
- ▶️ **Play** system media (Spotify, iTunes, etc.) when commercials start
- ⏸️ **Pause** system media when game resumes

**How it works:**
- Uses Windows Media Control API (`VK_MEDIA_PLAY_PAUSE`)
- Works with any media player that responds to system media keys
- Same as pressing the play/pause button on your keyboard

**Important Setup Notes:**

1. **Start with media paused/stopped** - The detector assumes you're starting with the game on screen, so your media player (Spotify, etc.) should NOT be playing when you start the detector. The detector will play media during commercials and pause it during the game.

2. **Disable Chrome's Hardware Media Key Handling** - This is required for media control to work properly:

> **⚠️ IMPORTANT:** For media control to work properly, you must **disable "Hardware Media Key Handling"** in Chrome:
> 1. Open Chrome and navigate to `chrome://flags`
> 2. Search for "Hardware Media Key Handling"
> 3. Set it to **Disabled**
> 4. Restart Chrome
>
> Without this setting, Chrome may intercept media keys and apply them to your broadcast instead of your media player.

## 🏗️ Project Structure

```
root/
├── detector.py                     # Main detection script
├── config.py                       # Configuration settings
├── train_sports_classifier.py      # Model training script (replaces train_nfl_classifier.py)
├── requirements.txt                # Python dependencies
├── models/                         # Trained models directory
│   ├── nfl_classifier_best.pth    # NFL model
│   ├── nba_classifier_best.pth    # NBA model
│   └── mlb_classifier_best.pth    # MLB model (etc.)
├── data/                           # Training data
│   ├── train/
│   │   ├── game/                  # Game footage images
│   │   └── commercial/            # Commercial images
│   ├── val/                       # Validation data
│   └── test/                      # Test data
├── screenshot_extractor_tool/     # Video processing utility
└── transitions/                   # Saved transition screenshots
```

## 🧠 Model Training

### Preparing Training Data

1. **Collect Screenshots**
   ```bash
   cd screenshot_extractor_tool
   python video_screenshot_extractor.py your_video.mp4
   ```

2. **Organize Data**
   
   Organize your screenshots by sport in the `data/` directory:
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
   # Train NFL model (default)
   python train_sports_classifier.py --sport nfl
   
   # Train NBA model
   python train_sports_classifier.py --sport nba --epochs 15
   
   # Train with custom data directory
   python train_sports_classifier.py --sport mlb --data-dir my_mlb_data --batch-size 64
   ```
   
   The trained model will be saved to `models/{sport}_classifier_best.pth`

### Command-line Options for Training

- `--sport`: Sport to train for (e.g., nfl, nba, mlb, nhl). Default: nfl
- `--data-dir`: Base directory for training data. Default: data
- `--epochs`: Number of training epochs. Default: 10
- `--batch-size`: Batch size for training. Default: 32

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

- The detector automatically closes Chrome and restarts it
- If Chrome won't close, manually kill it: `taskkill /F /IM chrome.exe`
- Then run `python detector.py` again
- Make sure no other processes are using port 9222

**"Chrome tab capture failed"**

- Make sure the video is playing in Chrome
- Navigate to your stream in the Chrome window that opened
- Start the video and press play
- The detector captures from the active Chrome tab

**"Could not initialize audio control"**

- Audio control initializes when Chrome starts playing audio
- Make sure your stream is playing with sound
- Check that Chrome has permission to play audio in Windows
- The detector will retry audio initialization automatically

### Detection Issues

**High false positive rate**
- Increase `min_confidence` to 0.95 in `config.py`
- Increase `threshold` to 0.65-0.7
- Increase `cooldown_seconds` to 30

**Model not detecting transitions**
- Lower `min_confidence` to 0.80 in `config.py`
- Check if the correct model exists in `models/{sport}_classifier_best.pth`
- Verify video is playing in Chrome
- Make sure you trained a model for the sport you're trying to detect

### Performance Tips

- **GPU Acceleration**: Install CUDA-compatible PyTorch for faster inference
- **Check Interval**: Increase to 2-3 seconds in config if CPU usage is high
- **Fullscreen**: Use fullscreen video for best detection accuracy
- **Persistent Profile**: Chrome profile is saved, so you only need to log in once

### Clean Exit

When you stop the detector with `Ctrl+C`, it automatically:
- **Unmutes Chrome** (restores audio)
- **Pauses media** (if media control is enabled and commercial was playing)
- **Closes Chrome** (to free up resources)
- **Shows session summary** (runtime, transitions detected, etc.)

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
2. Search existing [Issues](https://github.com/scottn12/Sports-Broadcast-Commercial-Detector/issues)
3. Create a new issue with detailed information

---

**Note**: This tool is designed for personal use to automatically mute commercials during sports broadcasts. Ensure compliance with your local laws and streaming service terms of use.

````