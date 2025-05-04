import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QSizePolicy
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

from ESPCN import ESPCN

class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESPCN 360p YouTube 추론")
        self.resize(960, 540)  # 초기 크기만 설정 (창 크기 조절 가능)

        # Layouts
        self.mainLayout = QVBoxLayout()
        self.topLayout = QHBoxLayout()
        self.videoLayout = QVBoxLayout()

        # URL 입력창
        self.urlInput = QLineEdit(self)
        self.urlInput.setPlaceholderText("YouTube URL을 입력하세요...")
        self.startButton = QPushButton("시작", self)
        self.startButton.clicked.connect(self.StartStream)

        self.topLayout.addWidget(self.urlInput)
        self.topLayout.addWidget(self.startButton)

        # 영상 표시용 라벨
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 창 크기에 따라 라벨도 확장

        self.videoLayout.addWidget(self.label)

        self.mainLayout.addLayout(self.topLayout)
        self.mainLayout.addLayout(self.videoLayout)

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

        self.currentQImage = None  # 최신 원본 QImage 저장

    def StartStream(self):
        path = self.urlInput.text().strip()
        if not path:
            print("URL 또는 파일 경로를 입력해주세요.")
            return

        if self.capture is not None:
            self.capture.release()

        if path.startswith("http"):
            # 유튜브 URL이면 스트림 URL 가져오기
            streamUrl = self.GetYoutube360pStreamUrl(path)
            if not streamUrl:
                print("스트리밍 URL을 가져올 수 없습니다.")
                return
            self.capture = cv2.VideoCapture(streamUrl)
        else:
            # 로컬 파일이면 바로 사용
            self.capture = cv2.VideoCapture(path)

        self.timer.start(16)  # 약 60 FPS


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
        if self.capture is None:
            return

        ret, frame = self.capture.read()
        if not ret:
            return

        # YCrCb 분리
        yCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y = yCrCb[:, :, 0]
        cr = yCrCb[:, :, 1]
        cb = yCrCb[:, :, 2]

        # Y 채널 초해상도
        yTensor = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).float().cuda() / 255.0
        with torch.no_grad():
            outY = self.model(yTensor).clamp(0, 1)

        # Cr, Cb 채널 보간
        crTensor = torch.from_numpy(cr).unsqueeze(0).unsqueeze(0).float().cuda() / 255.0
        cbTensor = torch.from_numpy(cb).unsqueeze(0).unsqueeze(0).float().cuda() / 255.0
        outCr = F.interpolate(crTensor, scale_factor=self.scale, mode='bicubic', align_corners=False)
        outCb = F.interpolate(cbTensor, scale_factor=self.scale, mode='bicubic', align_corners=False)

        # Merge
        merged = torch.cat([outY, outCr, outCb], dim=1).squeeze(0).permute(1, 2, 0).clamp(0, 1)
        mergedNP = (merged.cpu().numpy() * 255).astype(np.uint8)
        result = cv2.cvtColor(mergedNP, cv2.COLOR_YCrCb2BGR)

        # OpenCV BGR -> RGB 변환
        rgbImage = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        qtImage = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)

        self.currentQImage = qtImage  # 원본 QImage 저장
        
        scaledPixmap = QPixmap.fromImage(self.currentQImage).scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(scaledPixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())
