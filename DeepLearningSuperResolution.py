import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import subprocess

from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLineEdit, QPushButton, QSizePolicy, QCheckBox, QSlider, 
                           QProgressBar, QFrame, QStyle)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

from ESPCN import ESPCN

# 슬라이더 클릭시 바로 점프하는 커스텀 QSlider
class JumpSlider(QSlider):
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            value = QStyle.sliderValueFromPosition(
                self.minimum(), self.maximum(),
                event.x(), self.width()
            )
            self.setValue(value)
            self.sliderMoved.emit(value)  # 즉시 점프 신호
            event.accept()
        else:
            super().mousePressEvent(event)

class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESPCN 360p YouTube 추론")
        self.resize(960, 640)
        # 최소 크기 설정 추가
        self.setMinimumSize(640, 480)

        # Layouts
        self.mainLayout = QVBoxLayout()
        self.topLayout = QHBoxLayout()
        self.videoLayout = QVBoxLayout()
        self.controlLayout = QVBoxLayout()
        self.seekLayout = QHBoxLayout()
        self.playControlLayout = QHBoxLayout()

        # URL 입력창
        self.urlInput = QLineEdit(self)
        self.urlInput.setPlaceholderText("YouTube URL 또는 파일 경로를 입력하세요...")
        self.startButton = QPushButton("시작", self)
        self.startButton.clicked.connect(self.StartStream)

        self.topLayout.addWidget(self.urlInput)
        self.topLayout.addWidget(self.startButton)

        # 추론 on/off
        self.onOffBox = QCheckBox("초해상도", self)
        self.topLayout.addWidget(self.onOffBox)

        # 영상 표시용 라벨 - 크기 정책 수정
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        # 크기 정책을 Expanding으로 유지하되, 최소 크기 힌트 설정
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.setMinimumSize(320, 240)  # 최소 크기 설정
        self.label.setScaledContents(False)  # 스케일된 콘텐츠 비활성화
        # 기본 배경 설정
        self.label.setStyleSheet("QLabel { background-color: black; }")
        self.videoLayout.addWidget(self.label)

        # 컨트롤 바 구분선
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.HLine)
        self.separator.setFrameShadow(QFrame.Sunken)

        # 슬라이더: JumpSlider로 교체
        self.seekSlider = JumpSlider(Qt.Horizontal)
        self.seekSlider.setMinimum(0)
        self.seekSlider.setMaximum(100)
        self.seekSlider.setValue(0)
        self.seekSlider.setEnabled(False)
        self.seekSlider.sliderPressed.connect(self.OnSeekStart)
        self.seekSlider.sliderReleased.connect(self.OnSeekEnd)
        self.seekSlider.valueChanged.connect(self.OnSeekValueChanged)
        self.seekSlider.sliderMoved.connect(self.OnSliderMoved)  # <- 클릭/드래그 시 즉시 점프

        # 시간 표시 라벨
        self.timeLabel = QLabel("00:00 / 00:00")
        self.timeLabel.setFixedWidth(100)

        self.seekLayout.addWidget(self.seekSlider)
        self.seekLayout.addWidget(self.timeLabel)

        # 재생/일시정지 버튼
        self.playPauseButton = QPushButton("일시정지", self)
        self.playPauseButton.clicked.connect(self.TogglePlayPause)
        self.playPauseButton.setEnabled(False)

        # 점프 버튼들
        self.jumpBackward = QPushButton("◀◀ 10초", self)
        self.jumpBackward.clicked.connect(lambda: self.JumpSeconds(-10))
        self.jumpBackward.setEnabled(False)

        self.jumpForward = QPushButton("10초 ▶▶", self)
        self.jumpForward.clicked.connect(lambda: self.JumpSeconds(10))
        self.jumpForward.setEnabled(False)

        self.playControlLayout.addWidget(self.jumpBackward)
        self.playControlLayout.addWidget(self.playPauseButton)
        self.playControlLayout.addWidget(self.jumpForward)
        self.playControlLayout.addStretch()

        self.controlLayout.addWidget(self.separator)
        self.controlLayout.addLayout(self.seekLayout)
        self.controlLayout.addLayout(self.playControlLayout)

        self.mainLayout.addLayout(self.topLayout)
        self.mainLayout.addLayout(self.videoLayout)
        self.mainLayout.addLayout(self.controlLayout)

        self.setLayout(self.mainLayout)

        # 모델 초기화
        self.model = ESPCN(3)
        self.model.load_state_dict(torch.load("ESPCN.pth", map_location="cuda", weights_only=True))
        self.model = self.model.cuda()
        self.model.eval()

        self.scale = 3
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.UpdateFrame)

        self.currentQImage = None

        # 동영상 정보
        self.totalFrames = 0
        self.fps = 30
        self.currentFrame = 0
        self.isPlaying = False
        self.isSeeking = False
        self.duration = 0

    def StartStream(self):
        path = self.urlInput.text().strip()
        if not path:
            print("URL 또는 파일 경로를 입력해주세요.")
            return

        if self.capture is not None:
            self.capture.release()

        if path.startswith("http"):
            streamUrl = self.GetYoutube360pStreamUrl(path)
            if not streamUrl:
                print("스트리밍 URL을 가져올 수 없습니다.")
                return
            self.capture = cv2.VideoCapture(streamUrl)
        else:
            self.capture = cv2.VideoCapture(path)

        if self.capture.isOpened():
            self.totalFrames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 30
            self.duration = self.totalFrames / self.fps if self.totalFrames > 0 else 0

            if self.totalFrames <= 0:
                self.seekSlider.setEnabled(False)
                self.jumpBackward.setEnabled(False)
                self.jumpForward.setEnabled(False)
            else:
                self.seekSlider.setMaximum(self.totalFrames - 1)
                self.seekSlider.setEnabled(True)
                self.jumpBackward.setEnabled(True)
                self.jumpForward.setEnabled(True)

            self.playPauseButton.setEnabled(True)
            self.currentFrame = 0
            self.isPlaying = True
            self.timer.start(int(1000 / self.fps))
        else:
            print("동영상을 열 수 없습니다.")

    def GetYoutube360pStreamUrl(self, youtubeUrl):
        try:
            result = subprocess.run(
                ["yt-dlp", "-f", "best[height=360]", "-g", youtubeUrl],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print("yt-dlp 오류:", e.stderr.strip())
            return ""

    def UpdateFrame(self):
        if self.capture is None or not self.isPlaying:
            return

        ret, frame = self.capture.read()
        if not ret:
            self.isPlaying = False
            self.playPauseButton.setText("재생")
            return

        if self.totalFrames > 0:
            self.currentFrame = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
            if not self.isSeeking:
                self.seekSlider.setValue(self.currentFrame)
        else:
            self.currentFrame += 1

        self.UpdateTimeDisplay()

        if self.onOffBox.isChecked():
            yCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            y = yCrCb[:, :, 0]
            cr = yCrCb[:, :, 1]
            cb = yCrCb[:, :, 2]

            yTensor = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).float().cuda() / 255.0
            with torch.no_grad():
                outY = self.model(yTensor).clamp(0, 1)

            crTensor = torch.from_numpy(cr).unsqueeze(0).unsqueeze(0).float().cuda() / 255.0
            cbTensor = torch.from_numpy(cb).unsqueeze(0).unsqueeze(0).float().cuda() / 255.0
            outCr = F.interpolate(crTensor, scale_factor=self.scale, mode='bicubic', align_corners=False)
            outCb = F.interpolate(cbTensor, scale_factor=self.scale, mode='bicubic', align_corners=False)

            merged = torch.cat([outY, outCr, outCb], dim=1).squeeze(0).permute(1, 2, 0).clamp(0, 1)
            mergedNP = (merged.cpu().numpy() * 255).astype(np.uint8)
            result = cv2.cvtColor(mergedNP, cv2.COLOR_YCrCb2BGR)
            rgbImage = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        else:
            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.DisplayFrame(rgbImage)

    def DisplayFrame(self, rgbImage):
        """프레임을 화면에 표시하는 메서드 - 중복 코드 제거 및 크기 조정 개선"""
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        qtImage = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        
        self.currentQImage = qtImage
        
        # 라벨의 현재 크기에 맞춰 적절히 스케일링
        labelSize = self.label.size()
        scaledPixmap = QPixmap.fromImage(self.currentQImage).scaled(
            labelSize, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.label.setPixmap(scaledPixmap)

    def resizeEvent(self, event):
        """창 크기 변경 시 비디오 프레임 다시 스케일링"""
        super().resizeEvent(event)
        if self.currentQImage is not None:
            labelSize = self.label.size()
            scaledPixmap = QPixmap.fromImage(self.currentQImage).scaled(
                labelSize, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.label.setPixmap(scaledPixmap)

    def UpdateTimeDisplay(self):
        currentTime = self.currentFrame / self.fps
        totalTime = self.duration

        currentTimeStr = self.FormatTime(currentTime)
        totalTimeStr = self.FormatTime(totalTime) if totalTime > 0 else "∞"

        self.timeLabel.setText(f"{currentTimeStr} / {totalTimeStr}")

    def FormatTime(self, seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def TogglePlayPause(self):
        if self.capture is None:
            return

        self.isPlaying = not self.isPlaying
        if self.isPlaying:
            self.playPauseButton.setText("일시정지")
            self.timer.start(int(1000 / self.fps))
        else:
            self.playPauseButton.setText("재생")
            self.timer.stop()

    def JumpSeconds(self, seconds):
        if self.capture is None or self.totalFrames <= 0:
            return

        targetFrame = self.currentFrame + (seconds * self.fps)
        targetFrame = max(0, min(targetFrame, self.totalFrames - 1))

        self.SeekToFrame(int(targetFrame))

    def OnSeekStart(self):
        self.isSeeking = True

    def OnSeekEnd(self):
        if self.capture is None or self.totalFrames <= 0:
            return

        targetFrame = self.seekSlider.value()
        self.SeekToFrame(targetFrame)
        self.isSeeking = False

    def OnSeekValueChanged(self, value):
        if self.isSeeking and self.totalFrames > 0:
            currentTime = value / self.fps
            totalTime = self.duration
            currentTimeStr = self.FormatTime(currentTime)
            totalTimeStr = self.FormatTime(totalTime)
            self.timeLabel.setText(f"{currentTimeStr} / {totalTimeStr}")

    # 슬라이더 클릭·드래그로 이동
    def OnSliderMoved(self, value):
        if self.capture is not None and self.totalFrames > 0:
            self.SeekToFrame(value)

    def SeekToFrame(self, frameNumber):
        if self.capture is None:
            return

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frameNumber)
        self.currentFrame = frameNumber
        self.seekSlider.setValue(frameNumber)

        if not self.isPlaying:
            ret, frame = self.capture.read()
            if ret:
                if self.onOffBox.isChecked():
                    yCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                    y = yCrCb[:, :, 0]
                    cr = yCrCb[:, :, 1]
                    cb = yCrCb[:, :, 2]

                    yTensor = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).float().cuda() / 255.0
                    with torch.no_grad():
                        outY = self.model(yTensor).clamp(0, 1)

                    crTensor = torch.from_numpy(cr).unsqueeze(0).unsqueeze(0).float().cuda() / 255.0
                    cbTensor = torch.from_numpy(cb).unsqueeze(0).unsqueeze(0).float().cuda() / 255.0
                    outCr = F.interpolate(crTensor, scale_factor=self.scale, mode='bicubic', align_corners=False)
                    outCb = F.interpolate(cbTensor, scale_factor=self.scale, mode='bicubic', align_corners=False)

                    merged = torch.cat([outY, outCr, outCb], dim=1).squeeze(0).permute(1, 2, 0).clamp(0, 1)
                    mergedNP = (merged.cpu().numpy() * 255).astype(np.uint8)
                    result = cv2.cvtColor(mergedNP, cv2.COLOR_YCrCb2BGR)
                    rgbImage = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                else:
                    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                self.DisplayFrame(rgbImage)
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, frameNumber)

        self.UpdateTimeDisplay()

    def closeEvent(self, event):
        if self.capture:
            self.capture.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())