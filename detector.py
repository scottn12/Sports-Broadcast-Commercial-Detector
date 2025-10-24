"""
Sports Broadcast Commercial Detector
Monitors Chrome browser tab and detects when broadcast transitions between game and commercials
Automatically mutes Chrome during commercials (Windows)
Uses Chrome's tab capture API via Selenium for reliable screenshot capture
Supports multiple sports (NFL, NBA, MLB, NHL, etc.) via different trained models
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import time
from datetime import datetime
import os
from collections import deque
from config import DEFAULT_SPORT, DETECTION_CONFIG, RUNTIME_CONFIG
from pycaw.pycaw import AudioUtilities
import subprocess
import base64
import io
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import socket
import argparse


# Model setup
def create_model():
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )
    return model


# Image preprocessing
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class SportsTransitionDetector:
    def __init__(
        self,
        model_path,
        sport,
        threshold,
        cooldown_seconds,
        check_interval,
        smoothing_window,
        auto_mute,
        min_confidence,
        control_media,
        browser="chrome",
        chrome_debug_port=9222,
    ):
        """
        Initialize the sports broadcast transition detector

        Args:
            model_path: Path to trained model (.pth file)
            sport: Sport name (e.g., 'nfl', 'nba', 'mlb') for display purposes
            threshold: Confidence threshold for classification (0-1)
            cooldown_seconds: Minimum time between state transitions
            check_interval: Seconds between screen captures
            smoothing_window: Number of predictions to average for stability
            auto_mute: Automatically mute browser during commercials (Windows only)
            min_confidence: Minimum confidence required to trigger transition (default 0.65)
            control_media: Play system media (Spotify, etc.) during commercials, pause during game
            chrome_debug_port: Port for Chrome remote debugging (default 9222)
        """

        self.sport = sport.upper()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {self.sport.upper()} model on {self.device}...")

        self.model = create_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        # Configuration
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.check_interval = check_interval
        self.smoothing_window = smoothing_window
        self.auto_mute = auto_mute
        self.browser = browser
        self.min_confidence = min_confidence
        self.control_media = control_media
        self.chrome_debug_port = chrome_debug_port
        self.save_transitions = True  # Default, can be overridden in run()

        # State tracking
        self.current_state = None  # "game" or "commercial"
        self.last_transition_time = 0
        self.prediction_history = deque(maxlen=smoothing_window)

        # Statistics
        self.transition_count = 0
        self.start_time = time.time()
        self.state_start_time = None

        # Selenium WebDriver for Chrome tab capture
        self.driver = None
        try:
            self.driver = self._setup_chrome_driver()
            print(f"✓ Chrome WebDriver initialized (tab capture enabled)")
        except Exception as e:
            print(f"⚠️  Could not initialize Chrome WebDriver: {e}")
            print("   Falling back to screen capture method")
            self.driver = None

        # Audio control setup (Windows) - browser-specific
        self.browser_audio_session = None
        if self.auto_mute:
            # Don't initialize audio control yet - will do it when stream starts playing
            # Chrome needs to be playing audio before we can control it
            print(f"✓ Audio control will be initialized when stream starts playing")

        print("✓ Model loaded successfully!")
        print(f"✓ Using device: {self.device}")
        print(
            f"✓ Capture method: {'Chrome Tab Capture API' if self.driver else 'Screen Capture'}"
        )
        print(f"✓ Cooldown period: {cooldown_seconds} seconds")
        print(f"✓ Check interval: {check_interval} seconds")
        print(f"✓ Smoothing window: {smoothing_window} predictions")
        print(f"✓ Minimum confidence: {min_confidence:.0%}")
        print(f"✓ Auto-mute commercials: {'Yes' if self.auto_mute else 'No'}")
        print(f"✓ Control media playback: {'Yes' if self.control_media else 'No'}\n")

    def _setup_chrome_driver(self):
        """
        Setup Selenium Chrome WebDriver for tab capture
        Automatically starts Chrome in debug mode if not already running

        Returns:
            WebDriver instance
        """
        options = webdriver.ChromeOptions()

        self._start_chrome_debug_mode()
        # Wait for Chrome to start and become available
        import time

        print(f"   Waiting for Chrome to start...")
        max_wait = 20  # seconds - increased timeout
        waited = 0
        chrome_started = False
        while waited < max_wait:
            time.sleep(1)
            waited += 1
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", self.chrome_debug_port)) == 0:
                sock.close()
                chrome_started = True
                print(f"   Chrome is ready! (took {waited}s)")
                break
            sock.close()
            if waited % 5 == 0:
                print(f"   Still waiting for Chrome... ({waited}/{max_wait}s)")

        if not chrome_started:
            raise TimeoutError(
                f"Chrome failed to start in debug mode within {max_wait} seconds. "
                "Try closing all Chrome instances manually and running again."
            )

        # Connect to existing Chrome instance with remote debugging enabled
        options.add_experimental_option(
            "debuggerAddress", f"127.0.0.1:{self.chrome_debug_port}"
        )
        print(f"   Connecting to Chrome on port {self.chrome_debug_port}...")

        # Additional options for better performance
        options.add_argument("--disable-blink-features=AutomationControlled")
        # Set page load strategy to prevent hanging on slow pages
        options.page_load_strategy = "eager"

        try:
            # Use webdriver-manager to automatically download and manage ChromeDriver
            service = Service(ChromeDriverManager().install())

            # Create driver with timeout
            print(f"   Initializing WebDriver (this may take a moment)...")
            driver = webdriver.Chrome(service=service, options=options)

            # Set timeouts to prevent hanging
            driver.set_page_load_timeout(30)
            driver.set_script_timeout(30)

            # Make sure we have at least one window/tab
            if len(driver.window_handles) == 0:
                driver.switch_to.new_window("tab")

            # Navigate to a simple page that will stay open
            # User can open their stream in a new tab, or navigate this tab to their stream
            try:
                driver.get("about:blank")
            except:
                pass  # Ignore navigation errors

            return driver
        except WebDriverException as e:
            if self.use_existing_chrome:
                print(f"\n⚠️  Could not connect to Chrome.")
                print(f"   Error: {e}")
                print(f"   Chrome should have been started automatically.\n")
            raise

    def _start_chrome_debug_mode(self):
        """
        Start Chrome in debug mode with remote debugging enabled
        Closes existing Chrome instances and starts with your default profile
        """
        import subprocess
        import os
        import time

        # Common Chrome installation paths
        chrome_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(
                r"%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe"
            ),
        ]

        # Find Chrome
        chrome_path = None
        for path in chrome_paths:
            if os.path.exists(path):
                chrome_path = path
                break

        if not chrome_path:
            raise FileNotFoundError(
                "Chrome not found. Please install Google Chrome or specify the path manually."
            )

        print(f"   Closing any existing Chrome instances...")
        # Close all Chrome instances to allow starting with debug port
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", "chrome.exe"], capture_output=True, timeout=5
            )
        except:
            print("   ⚠️  Could not close existing Chrome instances.")

        # Start Chrome with debug port using a PERSISTENT separate profile
        # This profile will save your logins, so you only need to sign in once
        debug_profile_dir = os.path.join(
            os.path.expanduser("~"), "chrome-sports-detector"
        )

        chrome_args = [
            chrome_path,
            f"--remote-debugging-port={self.chrome_debug_port}",
            f"--user-data-dir={debug_profile_dir}",
        ]

        print(f"   Starting Chrome from: {chrome_path}")
        print(f"   Profile directory: {debug_profile_dir}")

        # Start Chrome in background
        if os.name == "nt":  # Windows
            subprocess.Popen(
                chrome_args,
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )

    def _find_browser_audio_session(self):
        """
        Find browser's audio session for per-application volume control

        Returns:
            Audio session object or None if not found
        """
        # Map browser names to executable names
        browser_executables = {
            "firefox": "firefox.exe",
            "chrome": "chrome.exe",
            "edge": "msedge.exe",
        }

        target_exe = browser_executables.get(self.browser)
        if not target_exe:
            print(f"⚠️  Unknown browser: {self.browser}")
            return None

        sessions = AudioUtilities.GetAllSessions()
        for session in sessions:
            if session.Process and session.Process.name().lower() == target_exe.lower():
                return session
        return None

    def _ensure_audio_control(self):
        """
        Ensure audio control is initialized. Call this when stream starts playing.
        Returns True if audio control is available, False otherwise.
        """
        if not self.auto_mute:
            return False

        if self.browser_audio_session:
            return True  # Already initialized

        # Try to find the audio session
        try:
            self.browser_audio_session = self._find_browser_audio_session()
            if self.browser_audio_session:
                print(f"   ✓ {self.browser} audio control initialized")
                return True
            else:
                # Only warn once
                if not hasattr(self, "_audio_warning_shown"):
                    print(
                        f"   ⚠️  {self.browser} audio not found - make sure stream is playing"
                    )
                    self._audio_warning_shown = True
                return False
        except Exception as e:
            if not hasattr(self, "_audio_error_shown"):
                print(f"   ⚠️  Could not initialize audio control: {e}")
                self._audio_error_shown = True
            return False

    def set_mute(self, mute):
        """
        Mute or unmute browser's audio

        Args:
            mute: True to mute, False to unmute
        """
        # Ensure audio control is initialized
        if not self._ensure_audio_control():
            return

        try:
            volume = self.browser_audio_session.SimpleAudioVolume
            volume.SetMute(1 if mute else 0, None)
            status = "🔇 MUTED" if mute else "🔊 UNMUTED"
            print(f"   {self.browser} {status}")
        except Exception as e:
            print(f"   ⚠️  {self.browser} audio control failed: {e}")
            # Reset audio session so we try to re-initialize next time
            self.browser_audio_session = None

    def get_mute_status(self):
        """
        Get current browser mute status

        Returns:
            True if muted, False if not, None if unavailable
        """
        if not self.browser_audio_session:
            return None

        try:
            volume = self.browser_audio_session.SimpleAudioVolume
            return bool(volume.GetMute())
        except:
            return None

    def toggle_media(self):
        """
        Send play command to Windows media (works with Spotify, etc.)
        Uses Windows Media Control API - sends VK_MEDIA_PLAY_PAUSE
        """
        if not self.control_media:
            return

        try:
            # 0xB3 = VK_MEDIA_PLAY_PAUSE virtual key code
            ps_command = """
            $wshell = New-Object -ComObject wscript.shell
            $wshell.SendKeys([char]0xB3)
            """
            subprocess.run(
                ["powershell", "-Command", ps_command],
                capture_output=True,
                timeout=3,
                check=True,
            )
            if self.current_state == "commercial":
                print(f"   ▶️  Media played")
            else:
                print(f"   ⏸️  Media paused")
        except Exception as e:
            print(f"   ⚠️  Could not play media: {e}")

    def capture_screen(self):
        """
        Capture screenshot using Chrome tab capture API

        Returns:
            PIL Image
        """
        if not self.driver:
            raise RuntimeError(
                "Chrome WebDriver not initialized. Cannot capture screenshot."
            )

        try:
            # Make sure we're capturing the active window
            # Try to find the window with actual content (not about:blank)
            try:
                current_url = self.driver.current_url
                # If we're on about:blank and there are other windows, switch to the last one
                if current_url == "about:blank" and len(self.driver.window_handles) > 1:
                    self.driver.switch_to.window(self.driver.window_handles[-1])
            except:
                pass  # Ignore window switching errors

            # Capture the current tab using Chrome's screenshot API
            screenshot_base64 = self.driver.get_screenshot_as_base64()

            if screenshot_base64 is None:
                raise ValueError("Screenshot returned None")

            # Decode base64 to image
            screenshot_bytes = base64.b64decode(screenshot_base64)
            image = Image.open(io.BytesIO(screenshot_bytes))

            return image
        except Exception as e:
            # Only print error once per session to avoid spam
            if not hasattr(self, "_chrome_error_shown"):
                error_msg = str(e).split("\n")[0]  # Just first line
                print(f"\n⚠️  Chrome tab capture failed: {error_msg}")
                print("   Possible causes:")
                print("   - No Chrome tab is open")
                print("   - Chrome window was closed")
                print("   - Navigate to your stream and try again\n")
                self._chrome_error_shown = True
            raise

    def classify_image(self, image):
        """
        Classify a PIL image

        Returns:
            (state, confidence, raw_probability)
            state: "game" or "commercial"
        """
        # Preprocess
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            output = self.model(image_tensor).item()

        # Determine state
        if output > self.threshold:
            state = "game"
            confidence = output
        else:
            state = "commercial"
            confidence = 1 - output

        return state, confidence, output

    def get_smoothed_prediction(self, current_prediction):
        """
        Average recent predictions for stability

        Returns:
            Smoothed state and confidence
        """
        self.prediction_history.append(current_prediction)

        if len(self.prediction_history) < self.smoothing_window:
            # Not enough history yet, return current
            return current_prediction

        # Average the probabilities
        avg_prob = sum(p[2] for p in self.prediction_history) / len(
            self.prediction_history
        )

        if avg_prob > self.threshold:
            state = "game"
            confidence = avg_prob
        else:
            state = "commercial"
            confidence = 1 - avg_prob

        return state, confidence, avg_prob

    def check_transition(self, new_state, confidence):
        """
        Check if we should transition to a new state

        Returns:
            True if transition occurred
        """
        current_time = time.time()

        # First detection (initialize state)
        if self.current_state is None:
            self.current_state = new_state
            self.last_transition_time = current_time
            self.state_start_time = current_time
            print(
                f"🎬 Initial state: {new_state.upper()} (confidence: {confidence:.1%})"
            )

            # Set initial audio state
            if self.auto_mute:
                if new_state == "commercial":
                    self.set_mute(True)
                else:
                    self.set_mute(False)

            # Set initial media state
            # We assume the media was off to start so we only toggle it on if the initial state is commercial
            if self.control_media and new_state == "commercial":
                self.toggle_media()

            return False

        # Check if state changed
        if new_state != self.current_state:
            # Check confidence threshold
            if confidence < self.min_confidence:
                print(f"  ⚠️  Low confidence ({confidence:.1%}) - ignoring transition")
                return False

            # Check cooldown period
            time_since_last_transition = current_time - self.last_transition_time

            if time_since_last_transition < self.cooldown_seconds:
                # Still in cooldown, ignore transition
                remaining = self.cooldown_seconds - time_since_last_transition
                print(
                    f"  ⏳ Cooldown active ({remaining:.0f}s remaining) - ignoring transition"
                )
                return False

            # Valid transition!
            old_state = self.current_state
            state_duration = current_time - self.state_start_time

            self.current_state = new_state
            self.last_transition_time = current_time
            self.state_start_time = current_time
            self.transition_count += 1

            # Log transition
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n{'='*70}")
            print(f"🔄 TRANSITION DETECTED at {timestamp}")
            print(f"   {old_state.upper()} → {new_state.upper()}")
            print(f"   Previous state duration: {state_duration:.1f} seconds")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Total transitions: {self.transition_count}")

            # Handle audio muting
            if self.auto_mute:
                if new_state == "commercial":
                    self.set_mute(True)
                elif new_state == "game":
                    self.set_mute(False)

            # Handle media playback
            if self.control_media:
                self.toggle_media()

            print(f"{'='*70}\n")

            return True

        return False

    def save_transition_screenshot(self, image, old_state, new_state):
        """Save screenshot when transition occurs"""
        if not self.save_transitions:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transitions/{timestamp}_{old_state}_to_{new_state}.jpg"
        image.save(filename)
        print(f"   📸 Screenshot saved: {filename}")

    def run(self, duration=None, save_transitions=True):
        """
        Main detection loop

        Args:
            duration: Optional runtime limit in seconds (None = run forever)
            save_transitions: Determines if we should save transition screenshots for this run
        """

        self.save_transitions = save_transitions

        # Create transitions directory if needed
        if self.save_transitions:
            os.makedirs("transitions", exist_ok=True)
        print("=" * 70)
        print(f"{self.sport} BROADCAST TRANSITION DETECTOR")
        print("=" * 70)
        print("Monitoring Chrome tab for transitions...")
        print("Press Ctrl+C to stop\n")

        print(f"\nStarting in 3 seconds...")
        time.sleep(3)
        print("🔴 MONITORING ACTIVE\n")

        iteration = 0

        try:
            while True:
                iteration += 1
                loop_start = time.time()

                # Check duration limit
                if duration and (loop_start - self.start_time) > duration:
                    print("\n⏰ Duration limit reached")
                    break

                # Capture screen
                screenshot = self.capture_screen()

                # Classify
                state, confidence, raw_prob = self.classify_image(screenshot)

                # Apply smoothing
                smoothed_state, smoothed_conf, smoothed_prob = (
                    self.get_smoothed_prediction((state, confidence, raw_prob))
                )

                # Check for transition
                transition_occurred = self.check_transition(
                    smoothed_state, smoothed_conf
                )

                if transition_occurred:
                    # Save screenshot if transition just happened
                    old_state = (
                        "game" if smoothed_state == "commercial" else "commercial"
                    )
                    self.save_transition_screenshot(
                        screenshot, old_state, smoothed_state
                    )

                # Status update
                elapsed = time.time() - self.state_start_time
                time_in_state = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"

                mute_status = ""
                if self.auto_mute:
                    is_muted = self.get_mute_status()
                    if is_muted is not None:
                        mute_status = (
                            f" | Audio: {'🔇 MUTED' if is_muted else '🔊 UNMUTED'}"
                        )

                status_line = (
                    f"[{iteration:04d}] Current: {self.current_state.upper():11s} | "
                    f"Duration: {time_in_state} | "
                    f"Confidence: {smoothed_conf:5.1%} | "
                    f"Transitions: {self.transition_count}{mute_status}"
                )

                print(f"\r{status_line}", end="", flush=True)

                # Sleep until next check
                elapsed_iteration = time.time() - loop_start
                sleep_time = max(0, self.check_interval - elapsed_iteration)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\n⏹ Stopped by user")

    def cleanup(self):
        """Cleanup resources (WebDriver, audio, etc.)"""
        # Pause media on exit
        if self.control_media and self.current_state == "commercial":
            print("Pausing media...")
            self.toggle_media()

        # Unmute browser on exit
        if self.auto_mute:
            print("Unmuting browser...")
            self.set_mute(False)

        print("Closing chrome...")
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", "chrome.exe"],
                capture_output=True,
                timeout=5,
            )
        except:
            print("   ⚠️  Could not close existing Chrome instances.")

        self.print_summary()

    def print_summary(self):
        """Print session statistics"""
        total_runtime = time.time() - self.start_time

        print("\n")
        print("=" * 70)
        print("SESSION SUMMARY")
        print("=" * 70)
        print(
            f"Total runtime: {int(total_runtime//60)} minutes {int(total_runtime%60)} seconds"
        )
        print(f"Total transitions detected: {self.transition_count}")
        print(
            f"Final state: {self.current_state.upper() if self.current_state else 'Unknown'}"
        )

        if self.transition_count > 0:
            avg_time_between = total_runtime / self.transition_count
            print(f"Average time between transitions: {avg_time_between:.1f} seconds")

        if self.save_transitions:
            print(f"\nTransition screenshots saved in: ./transitions/")

        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sports Broadcast Commercial Detector")
    parser.add_argument(
        "--sport",
        type=str,
        default=DEFAULT_SPORT,
        help=f"Sport to detect (e.g., nfl, nba, mlb, nhl). Default: {DEFAULT_SPORT}",
    )
    args = parser.parse_args()
    sport = args.sport.lower()
    model_path = os.path.join("models", f"{sport.lower()}_classifier_best.pth")

    print("=" * 70)
    print(f"{sport.upper()} DETECTOR STARTING")
    print("=" * 70)
    print("\nAny existing chrome tabs will be closed.")
    print("A new chrome session will start automatically in debug mode.")
    print(f"After Chrome opens, navigate to your {sport.upper()} stream.")
    print("-" * 70)

    # Initialize detector with Chrome tab capture
    detector = SportsTransitionDetector(
        model_path=model_path, sport=sport, **DETECTION_CONFIG
    )

    # Give user time to navigate to their stream
    aborted = False
    if detector.driver:
        print("\n" + "=" * 70)
        print(f"{sport} DETECTOR IS READY!")
        print("=" * 70)
        print("\n📺 FINAL STEPS:")
        print(
            '1. A Chrome window should be open. Wait for it to navigate to "about:blank".'
        )
        print(f"2. Navigate to your {sport.upper()} stream in that Chrome tab.")
        print("3. Start the video playing.")
        print("4. You should use fullscreen for best results.")
        print("5. Come back here and press ENTER to start detection.")
        print("\n💡 TIP: The detector captures directly from the first Chrome tab.")
        print("   Keep the first Chrome tab open during detection!")
        print("=" * 70)
        try:
            input("\nPress ENTER when your stream is playing and ready...")
        except:
            aborted = True

    # Run detector
    try:
        if not aborted:
            detector.run(**RUNTIME_CONFIG)
    finally:
        detector.cleanup()
