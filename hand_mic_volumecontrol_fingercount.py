"""
hand_mic_volume_finger_absolute_final.py

FINAL VERSION:
- Finger count maps directly to MIC VOLUME (0–100%)
- 0 fingers = 0%
- 1 finger = 20%
- 2 = 40%
- 3 = 60%
- 4 = 80%
- 5 = 100%

- Volume DECREASES when finger count decreases
- Volume INCREASES when finger count increases
- Pinch gesture toggles mute
- PyQt5 UI + threaded MediaPipe + threaded Audio
"""

import sys
import time
import math
import queue

import cv2
import numpy as np
import mediapipe as mp

from PyQt5 import QtCore, QtGui, QtWidgets

# ---------------- AUDIO CONTROL (MIC VOLUME) ----------------
from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL, CoInitialize, CoUninitialize
from comtypes.client import CreateObject
from comtypes import GUID
from pycaw.pycaw import IAudioEndpointVolume, IMMDeviceEnumerator


# ---------------- CONFIG ----------------
PINCH_THRESHOLD = 0.03
PINCH_COOLDOWN = 1.0
CAM_INDEX = 0
FPS = 30


# ------------------------------------------------------------
# AUDIO THREAD — SAFE COM THREAD
# ------------------------------------------------------------
class AudioThread(QtCore.QThread):
    state_updated = QtCore.pyqtSignal(float, bool)

    def __init__(self, cmd_queue: queue.Queue):
        super().__init__()
        self.cmd_queue = cmd_queue
        self.running = True

    def _get_mic_interface(self):
        # Use GUID fallback (works on every Windows build)
        clsid = GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
        enumerator = CreateObject(clsid, interface=IMMDeviceEnumerator)

        mic = enumerator.GetDefaultAudioEndpoint(1, 0)  # eCapture = 1
        iface = mic.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(iface, POINTER(IAudioEndpointVolume))

    def run(self):
        CoInitialize()
        try:
            mic = self._get_mic_interface()

            # Send initial state
            try:
                vol = float(mic.GetMasterVolumeLevelScalar()) * 100
                muted = bool(mic.GetMute())
                self.state_updated.emit(vol, muted)
            except:
                self.state_updated.emit(0, False)

            while self.running:
                try:
                    cmd, arg = self.cmd_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                if cmd == "set_volume":
                    pct = max(0, min(100, float(arg)))
                    mic.SetMasterVolumeLevelScalar(pct / 100.0, None)

                elif cmd == "toggle_mute":
                    mic.SetMute(0 if mic.GetMute() else 1, None)

                elif cmd == "refresh":
                    pass

                # Emit updated state
                try:
                    vol = float(mic.GetMasterVolumeLevelScalar()) * 100
                    muted = bool(mic.GetMute())
                    self.state_updated.emit(vol, muted)
                except:
                    pass

        finally:
            CoUninitialize()

    def stop(self):
        self.running = False



# ------------------------------------------------------------
# FRAME WORKER THREAD — MediaPipe Hands
# ------------------------------------------------------------
class FrameWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage, object)

    def __init__(self, cam_index=0):
        super().__init__()
        self.cam_index = cam_index
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.cam_index)
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=1)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            qimg = QtGui.QImage(
                rgb.data, rgb.shape[1], rgb.shape[0],
                rgb.strides[0], QtGui.QImage.Format_RGB888
            )

            self.frame_ready.emit(qimg.copy(), results)

        cap.release()

    def stop(self):
        self.running = False



# ------------------------------------------------------------
# MAIN WINDOW UI
# ------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Finger-Controlled MIC Volume (Final Version)")
        self.resize(1100, 650)

        # Layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Camera feed
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(800, 600)
        self.video_label.setStyleSheet("background:#000;")
        layout.addWidget(self.video_label)

        # Right panel
        panel = QtWidgets.QVBoxLayout()
        layout.addLayout(panel)

        self.vol_label = QtWidgets.QLabel("Mic Volume: --%")
        panel.addWidget(self.vol_label)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        panel.addWidget(self.progress)

        self.mute_label = QtWidgets.QLabel("Mic Muted: ?")
        panel.addWidget(self.mute_label)

        panel.addSpacing(20)
        panel.addWidget(QtWidgets.QLabel("Finger Mapping:"))
        panel.addWidget(QtWidgets.QLabel("0 fingers = 0%"))
        panel.addWidget(QtWidgets.QLabel("1 finger = 20%"))
        panel.addWidget(QtWidgets.QLabel("2 fingers = 40%"))
        panel.addWidget(QtWidgets.QLabel("3 fingers = 60%"))
        panel.addWidget(QtWidgets.QLabel("4 fingers = 80%"))
        panel.addWidget(QtWidgets.QLabel("5 fingers = 100%"))

        panel.addSpacing(10)
        panel.addWidget(QtWidgets.QLabel("Pinch = Mute/Unmute"))
        panel.addStretch()

        # State
        self.current_volume = 0
        self.muted = False
        self.last_finger_gesture = -1
        self.last_pinch_time = 0

        # Audio thread
        self.cmd_queue = queue.Queue()
        self.audio_thread = AudioThread(self.cmd_queue)
        self.audio_thread.state_updated.connect(self.update_audio_ui)
        self.audio_thread.start()

        # Camera thread
        self.worker = FrameWorker(cam_index=CAM_INDEX)
        self.worker.frame_ready.connect(self.process_frame)
        self.worker.start()

    # ---------------- FINGER COUNTING ----------------
    def count_fingers(self, lm, handedness):
        fingers = []

        # Thumb
        if handedness == "Right":
            fingers.append(lm[4].x < lm[3].x)
        else:
            fingers.append(lm[4].x > lm[3].x)

        # Other fingers
        fingers.append(lm[8].y < lm[6].y)
        fingers.append(lm[12].y < lm[10].y)
        fingers.append(lm[16].y < lm[14].y)
        fingers.append(lm[20].y < lm[18].y)

        return fingers.count(True)

    # ---------------- PER-FRAME PROCESSING ----------------
    @QtCore.pyqtSlot(QtGui.QImage, object)
    def process_frame(self, qimg, results):
        # Display video
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), QtCore.Qt.KeepAspectRatio
        )
        self.video_label.setPixmap(pix)

        if not results or not results.multi_hand_landmarks:
            return

        lm = results.multi_hand_landmarks[0].landmark
        handedness = results.multi_handedness[0].classification[0].label

        # Pinch → mute
        thumb, index = lm[4], lm[8]
        dist = math.hypot(thumb.x - index.x, thumb.y - index.y)

        if dist < PINCH_THRESHOLD and (time.time() - self.last_pinch_time) > PINCH_COOLDOWN:
            self.cmd_queue.put(("toggle_mute", None))
            self.last_pinch_time = time.time()
            return

        # -------- FINAL ABSOLUTE VOLUME CONTROL LOGIC --------
        finger_count = self.count_fingers(lm, handedness)
        target_volume = finger_count * 20  # absolute mapping

        if finger_count != self.last_finger_gesture:
            self.cmd_queue.put(("set_volume", target_volume))
            self.cmd_queue.put(("refresh", None))
            self.last_finger_gesture = finger_count

    # ---------------- UPDATE UI ----------------
    @QtCore.pyqtSlot(float, bool)
    def update_audio_ui(self, vol, muted):
        self.current_volume = vol
        self.mute_label.setText(f"Mic Muted: {muted}")
        self.vol_label.setText(f"Mic Volume: {vol:.0f}%")
        self.progress.setValue(int(vol))

    # ---------------- CLEAN EXIT ----------------
    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait()
        self.audio_thread.stop()
        self.audio_thread.wait()
        event.accept()



# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
