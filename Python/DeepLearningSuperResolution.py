import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yt_dlp

from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
                             QLineEdit, QPushButton, QSizePolicy, QCheckBox, QSlider,
                             QFrame, QStyle)
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
        self.setWindowTitle("ESPCN Patch 기반 실시간 초해상도")
        self.resize(960, 640)
        self.setMinimumSize(640, 480)

        self.MainLayout = QVBoxLayout()
        self.TopLayout = QHBoxLayout()
        self.VideoLayout = QVBoxLayout()
        self.ControlLayout = QVBoxLayout()
        self.SeekLayout = QHBoxLayout()
        self.PlayControlLayout = QHBoxLayout()

        # URL 입력창
        self.UrlInput = QLineEdit(self)
        self.UrlInput.setPlaceholderText("YouTube URL 또는 파일 경로를 입력하세요...")
        self.StartButton = QPushButton("시작", self)
        self.StartButton.clicked.connect(self.StartStream)
        self.TopLayout.addWidget(self.UrlInput)
        self.TopLayout.addWidget(self.StartButton)

        # 추론 on/off
        self.OnOffBox = QCheckBox("초해상도", self)
        self.TopLayout.addWidget(self.OnOffBox)

        # 영상 표시용 라벨
        self.Label = QLabel(self)
        self.Label.setAlignment(Qt.AlignCenter)
        self.Label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.Label.setMinimumSize(320, 240)
        self.Label.setStyleSheet("QLabel { background-color: black; }")
        self.VideoLayout.addWidget(self.Label)

        # 컨트롤 바 구분선
        self.Separator = QFrame()
        self.Separator.setFrameShape(QFrame.HLine)
        self.Separator.setFrameShadow(QFrame.Sunken)

        # 슬라이더
        self.SeekSlider = JumpSlider(Qt.Horizontal)
        self.SeekSlider.setMinimum(0)
        self.SeekSlider.setMaximum(100)
        self.SeekSlider.setValue(0)
        self.SeekSlider.setEnabled(False)
        self.SeekSlider.sliderPressed.connect(self.OnSeekStart)
        self.SeekSlider.sliderReleased.connect(self.OnSeekEnd)
        self.SeekSlider.valueChanged.connect(self.OnSeekValueChanged)
        self.SeekSlider.sliderMoved.connect(self.OnSliderMoved)

        # 시간 표시 라벨
        self.TimeLabel = QLabel("00:00 / 00:00")
        self.TimeLabel.setFixedWidth(100)

        self.SeekLayout.addWidget(self.SeekSlider)
        self.SeekLayout.addWidget(self.TimeLabel)

        # 재생/일시정지 버튼
        self.PlayPauseButton = QPushButton("일시정지", self)
        self.PlayPauseButton.clicked.connect(self.TogglePlayPause)
        self.PlayPauseButton.setEnabled(False)

        # 점프 버튼들
        self.JumpBackward = QPushButton("◀◀ 10초", self)
        self.JumpBackward.clicked.connect(lambda: self.JumpSeconds(-10))
        self.JumpBackward.setEnabled(False)

        self.JumpForward = QPushButton("10초 ▶▶", self)
        self.JumpForward.clicked.connect(lambda: self.JumpSeconds(10))
        self.JumpForward.setEnabled(False)

        self.PlayControlLayout.addWidget(self.JumpBackward)
        self.PlayControlLayout.addWidget(self.PlayPauseButton)
        self.PlayControlLayout.addWidget(self.JumpForward)
        self.PlayControlLayout.addStretch()

        self.ControlLayout.addWidget(self.Separator)
        self.ControlLayout.addLayout(self.SeekLayout)
        self.ControlLayout.addLayout(self.PlayControlLayout)

        self.MainLayout.addLayout(self.TopLayout)
        self.MainLayout.addLayout(self.VideoLayout)
        self.MainLayout.addLayout(self.ControlLayout)

        self.setLayout(self.MainLayout)

        # 모델 초기화
        self.Scale = 3
        self.PatchSize = 64  # 패치 단위 크기
        self.Model = ESPCN(self.Scale)
        self.Model.load_state_dict(torch.load("ESPCN.pth", map_location="cuda", weights_only=True))
        self.Model = self.Model.cuda()
        self.Model.eval()

        self.Capture = None
        self.Timer = QTimer()
        self.Timer.timeout.connect(self.UpdateFrame)
        self.CurrentQImage = None

        self.TotalFrames = 0
        self.Fps = 30
        self.CurrentFrame = 0
        self.IsPlaying = False
        self.IsSeeking = False
        self.Duration = 0

    def StartStream(self):
        path = self.UrlInput.text().strip()
        if not path:
            print("URL 또는 파일 경로를 입력해주세요.")
            return

        if self.Capture is not None:
            self.Capture.release()

        if path.startswith("http"):
            streamUrl = self.GetYoutubeStreamUrl(path)
            if not streamUrl:
                print("스트리밍 URL을 가져올 수 없습니다.")
                return
            self.Capture = cv2.VideoCapture(streamUrl)
        else:
            self.Capture = cv2.VideoCapture(path)

        if self.Capture.isOpened():
            self.TotalFrames = int(self.Capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.Fps = self.Capture.get(cv2.CAP_PROP_FPS) or 30
            self.Duration = self.TotalFrames / self.Fps if self.TotalFrames > 0 else 0

            if self.TotalFrames <= 0:
                self.SeekSlider.setEnabled(False)
                self.JumpBackward.setEnabled(False)
                self.JumpForward.setEnabled(False)
            else:
                self.SeekSlider.setMaximum(self.TotalFrames - 1)
                self.SeekSlider.setEnabled(True)
                self.JumpBackward.setEnabled(True)
                self.JumpForward.setEnabled(True)

            self.PlayPauseButton.setEnabled(True)
            self.CurrentFrame = 0
            self.IsPlaying = True
            self.Timer.start(int(1000 / self.Fps))
        else:
            print("동영상을 열 수 없습니다.")

    def GetYoutubeStreamUrl(self, youtubeUrl):
        options = {
            "quiet": True,
            "skip_download": True,
            "format": "bestvideo+bestaudio/best"
        }
        try:
            with yt_dlp.YoutubeDL(options) as youtubeDl:
                infoDict = youtubeDl.extract_info(youtubeUrl, download=False)
                # DASH/WEBM 등은 formats 리스트로 반환되므로 bestvideo+bestaudio 사용
                if "url" in infoDict:
                    return infoDict["url"]
                if "formats" in infoDict:
                    # 가장 해상도 높은 비디오 우선 반환
                    videoFormats = [f for f in infoDict["formats"] if f.get("vcodec", "none") != "none"]
                    videoFormats = sorted(videoFormats, key=lambda x: x.get("height", 0), reverse=True)
                    if videoFormats:
                        return videoFormats[0]["url"]
                    return infoDict["formats"][0]["url"]
        except Exception as ex:
            print("yt-dlp 오류:", str(ex))
            return ""

    def UpdateFrame(self):
        if self.Capture is None or not self.IsPlaying:
            return

        ret, frame = self.Capture.read()
        if not ret:
            self.IsPlaying = False
            self.PlayPauseButton.setText("재생")
            return

        if self.TotalFrames > 0:
            self.CurrentFrame = int(self.Capture.get(cv2.CAP_PROP_POS_FRAMES))
            if not self.IsSeeking:
                self.SeekSlider.setValue(self.CurrentFrame)
        else:
            self.CurrentFrame += 1

        self.UpdateTimeDisplay()

        if self.OnOffBox.isChecked():
            srResult = self.RunSuperResolutionByPatch(frame, self.PatchSize)
            rgbImage = cv2.cvtColor(srResult, cv2.COLOR_BGR2RGB)
        else:
            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.DisplayFrame(rgbImage)

    def RunSuperResolutionByPatch(self, bgrFrame, patchSize, overlap=32):
        scale = self.Scale
        yCrCb = cv2.cvtColor(bgrFrame, cv2.COLOR_BGR2YCrCb)
        y = yCrCb[:, :, 0]
        cr = yCrCb[:, :, 1]
        cb = yCrCb[:, :, 2]
        h, w = y.shape

        # 패딩 계산 (reflect 모드로 변경하여 경계 아티팩트 감소)
        padH = (patchSize - (h % patchSize)) % patchSize
        padW = (patchSize - (w % patchSize)) % patchSize
        yPad = np.pad(y, ((0, padH), (0, padW)), 'reflect')
        crPad = np.pad(cr, ((0, padH), (0, padW)), 'reflect')
        cbPad = np.pad(cb, ((0, padH), (0, padW)), 'reflect')
        newH, newW = yPad.shape

        step = max(patchSize - overlap, 8)  # step이 너무 작아지지 않도록 제한
        outH, outW = newH * scale, newW * scale

        # 결과 배열/가중치 배열 생성
        outY = np.zeros((outH, outW), dtype=np.float32)
        outWeights = np.zeros((outH, outW), dtype=np.float32)

        # 가우시안 가중치 마스크 생성 (더 효율적인 방법)
        center = patchSize * scale // 2
        
        # 테이퍼 윈도우 방식으로 변경 (더 빠름)
        taperSize = min(overlap * scale, patchSize * scale // 3)
        window = np.ones((patchSize * scale, patchSize * scale), dtype=np.float32)
        
        # 경계에서 선형 테이퍼 적용 (가우시안보다 빠름)
        for i in range(taperSize):
            weight = (i + 1) / taperSize
            window[i, :] *= weight
            window[-(i+1), :] *= weight
            window[:, i] *= weight
            window[:, -(i+1)] *= weight

        # 오버랩 블록 처리
        for row in range(0, newH - patchSize + 1, step):
            for col in range(0, newW - patchSize + 1, step):
                patch = yPad[row:row+patchSize, col:col+patchSize]
                patchTensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().cuda() / 255.0
                
                with torch.no_grad():
                    outPatch = self.Model(patchTensor).clamp(0, 1)
                
                outPatch = outPatch.squeeze().cpu().numpy()
                outRow = row * scale
                outCol = col * scale
                
                # 선형 테이퍼 윈도우 적용하여 부드러운 블렌딩
                outY[outRow:outRow+patchSize*scale, outCol:outCol+patchSize*scale] += outPatch * window
                outWeights[outRow:outRow+patchSize*scale, outCol:outCol+patchSize*scale] += window

        # 가중치로 정규화 (0으로 나누는 것 방지)
        outY = np.divide(outY, outWeights, out=outY, where=outWeights > 1e-8)

        # 패딩 제거
        outY = outY[:h*scale, :w*scale]

        # cr/cb 업스케일 (속도와 품질 균형)
        crUpscaled = cv2.resize(cr, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
        cbUpscaled = cv2.resize(cb, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

        # 결과 합성
        merged = np.stack([outY, crUpscaled/255.0, cbUpscaled/255.0], axis=2)
        merged = np.clip(merged * 255.0, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
        
        # 후처리: 가벼운 블러로 격자 제거 (속도 우선)
        result = cv2.GaussianBlur(result, (3, 3), 0.7)
        
        return result

    def DisplayFrame(self, rgbImage):
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        qtImage = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        self.CurrentQImage = qtImage
        labelSize = self.Label.size()
        scaledPixmap = QPixmap.fromImage(self.CurrentQImage).scaled(
            labelSize,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.Label.setPixmap(scaledPixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.CurrentQImage is not None:
            labelSize = self.Label.size()
            scaledPixmap = QPixmap.fromImage(self.CurrentQImage).scaled(
                labelSize,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.Label.setPixmap(scaledPixmap)

    def UpdateTimeDisplay(self):
        currentTime = self.CurrentFrame / self.Fps
        totalTime = self.Duration
        currentTimeStr = self.FormatTime(currentTime)
        totalTimeStr = self.FormatTime(totalTime) if totalTime > 0 else "∞"
        self.TimeLabel.setText(f"{currentTimeStr} / {totalTimeStr}")

    def FormatTime(self, seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def TogglePlayPause(self):
        if self.Capture is None:
            return

        self.IsPlaying = not self.IsPlaying
        if self.IsPlaying:
            self.PlayPauseButton.setText("일시정지")
            self.Timer.start(int(1000 / self.Fps))
        else:
            self.PlayPauseButton.setText("재생")
            self.Timer.stop()

    def JumpSeconds(self, seconds):
        if self.Capture is None or self.TotalFrames <= 0:
            return

        targetFrame = self.CurrentFrame + (seconds * self.Fps)
        targetFrame = max(0, min(targetFrame, self.TotalFrames - 1))
        self.SeekToFrame(int(targetFrame))

    def OnSeekStart(self):
        self.IsSeeking = True

    def OnSeekEnd(self):
        if self.Capture is None or self.TotalFrames <= 0:
            return

        targetFrame = self.SeekSlider.value()
        self.SeekToFrame(targetFrame)
        self.IsSeeking = False

    def OnSeekValueChanged(self, value):
        if self.IsSeeking and self.TotalFrames > 0:
            currentTime = value / self.Fps
            totalTime = self.Duration
            currentTimeStr = self.FormatTime(currentTime)
            totalTimeStr = self.FormatTime(totalTime)
            self.TimeLabel.setText(f"{currentTimeStr} / {totalTimeStr}")

    def OnSliderMoved(self, value):
        if self.Capture is not None and self.TotalFrames > 0:
            self.SeekToFrame(value)

    def SeekToFrame(self, frameNumber):
        if self.Capture is None:
            return

        self.Capture.set(cv2.CAP_PROP_POS_FRAMES, frameNumber)
        self.CurrentFrame = frameNumber
        self.SeekSlider.setValue(frameNumber)

        if not self.IsPlaying:
            ret, frame = self.Capture.read()
            if ret:
                if self.OnOffBox.isChecked():
                    srResult = self.RunSuperResolutionByPatch(frame, self.PatchSize)
                    rgbImage = cv2.cvtColor(srResult, cv2.COLOR_BGR2RGB)
                else:
                    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.DisplayFrame(rgbImage)
                self.Capture.set(cv2.CAP_PROP_POS_FRAMES, frameNumber)

        self.UpdateTimeDisplay()

    def closeEvent(self, event):
        if self.Capture:
            self.Capture.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())
