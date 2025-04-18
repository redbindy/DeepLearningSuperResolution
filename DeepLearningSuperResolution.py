import cv2
import numpy as np
import torch
import torch.nn.functional as F
import ESPCN

# model = ESPCN(3)
# model.load_state_dict(torch.load("ESPCN.pth", map_location = "cuda", weights_only = True))
# model = model.cuda()
# model.eval()

# capture360 = cv2.VideoCapture("./Data/TestBaseball360p.mp4")
# capture1080 = cv2.VideoCapture("./Data/TestBaseball1080p.mp4")

# cv2.namedWindow("SuperResolution", cv2.WINDOW_NORMAL)
# while True:
#     bSuccess360, frame360 = capture360.read()
#     bSuccess1080, frame1080 = capture1080.read()
    
#     bEnd = not bSuccess360 or not bSuccess1080
#     if bEnd:
#         break

#     targetSize = (frame1080.shape[:2])

#     yCrCb = cv2.cvtColor(frame360, cv2.COLOR_BGR2YCrCb)
#     y = yCrCb[:, :, 0]
#     cr = yCrCb[:, :, 1]
#     cb = yCrCb[:, :, 2]    

#     yTensor = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0) / 255.0
#     yTensor = yTensor.cuda()
    
#     with torch.no_grad():
#         outputY = model(yTensor).clamp(0.0, 1.0)

#     crTensor = torch.from_numpy(cr).unsqueeze(0).unsqueeze(0) / 255.0
#     cbTensor = torch.from_numpy(cb).unsqueeze(0).unsqueeze(0) / 255.0

#     crTensor = crTensor.cuda()
#     cbTensor = cbTensor.cuda()

#     resizedCr = F.interpolate(crTensor, size = targetSize, mode = "bicubic", align_corners = False)
#     resizedCb = F.interpolate(cbTensor, size = targetSize, mode = "bicubic", align_corners = False)

#     outputY = outputY.squeeze().cpu().numpy() * 255.0
#     outputY = outputY.astype(np.uint8)
#     crUp = (resizedCr.squeeze().clamp(0.0, 1.0) * 255).byte().cpu().numpy()
#     cbUp = (resizedCb.squeeze().clamp(0.0, 1.0) * 255).byte().cpu().numpy()

#     yCrCbUp = cv2.merge([outputY, crUp, cbUp])
#     rgbUpscaled = cv2.cvtColor(yCrCbUp, cv2.COLOR_YCrCb2BGR)

#     combined = np.hstack((frame1080, rgbUpscaled))

#     cv2.imshow("SuperResolution", combined)
        
#     if cv2.waitKey(33) & 0xFF == ord('q'):
#         break;

# cv2.destroyAllWindows()

# capture1080.release()
# capture360.release()

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
from ESPCN import ESPCN

def get_youtube_360p_stream_url(youtube_url: str) -> str:
    try:
        result = subprocess.run(
            ["yt-dlp", "-f", "best[height=360]", "-g", youtube_url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print("yt-dlp 오류:", e.stderr.strip())
        return ""

def run_espcn_on_youtube(youtube_url, scale=3):
    stream_url = get_youtube_360p_stream_url(youtube_url)
    if not stream_url:
        print("스트리밍 URL을 가져올 수 없습니다.")
        return

    model = ESPCN(3)
    model.load_state_dict(torch.load("ESPCN.pth", map_location = "cuda", weights_only = True))
    model = model.cuda()
    model.eval()

    cap = cv2.VideoCapture(stream_url)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 입력은 이미 360p로 가정함
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0]
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]

        y_tensor = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).float().cuda() / 255.0
        with torch.no_grad():
            out_y = model(y_tensor).clamp(0, 1)

        cr_tensor = torch.from_numpy(cr).unsqueeze(0).unsqueeze(0).float().cuda() / 255.0
        cb_tensor = torch.from_numpy(cb).unsqueeze(0).unsqueeze(0).float().cuda() / 255.0
        out_cr = F.interpolate(cr_tensor, scale_factor=scale, mode='bicubic', align_corners=False)
        out_cb = F.interpolate(cb_tensor, scale_factor=scale, mode='bicubic', align_corners=False)

        merged = torch.cat([out_y, out_cr, out_cb], dim=1).squeeze(0).permute(1, 2, 0).clamp(0, 1)
        merged_np = (merged.cpu().numpy() * 255).astype(np.uint8)
        result = cv2.cvtColor(merged_np, cv2.COLOR_YCrCb2BGR)

        cv2.imshow("ESPCN 360p YouTube 추론", result)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

run_espcn_on_youtube("https://www.youtube.com/watch?v=FZGXlyzSz3M")