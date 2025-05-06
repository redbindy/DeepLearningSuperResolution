package com.example.android

import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.Color
import android.media.MediaMetadataRetriever
import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.example.android.ui.theme.AndroidTheme
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.flex.FlexDelegate
import java.io.FileInputStream
import java.lang.ref.WeakReference
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            AndroidTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    VideoFrameUpscale()
                }
            }
        }
    }
}

@Composable
fun VideoFrameUpscale() {
    val context = LocalContext.current
    var selectedUri by remember { mutableStateOf<Uri?>(null) }
    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var isPlaying by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    // TFLite 모델 인터프리터 및 프레임 처리 컨트롤러
    val frameProcessor = remember { FrameProcessor() }
    val processingJob = remember { mutableStateOf<Job?>(null) }

    // 모델 로딩 처리
    LaunchedEffect(Unit) {
        try {
            isLoading = true
            frameProcessor.loadModel(context)
            isLoading = false
        } catch (e: Exception) {
            Log.e("VideoUpscale", "모델 로딩 실패", e)
            errorMessage = "모델 로딩 실패: ${e.localizedMessage}"
            isLoading = false
        }
    }

    // 비디오 선택기
    val getVideoUri = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        selectedUri = uri
        errorMessage = null
        if (uri != null) {
            isPlaying = true
        }
    }

    // 앱 종료시 리소스 정리
    DisposableEffect(Unit) {
        onDispose {
            processingJob.value?.cancel()
            frameProcessor.release()
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .clickable(enabled = !isLoading && !isPlaying) {
                getVideoUri.launch("video/*")
            },
        contentAlignment = Alignment.Center
    ) {
        when {
            isLoading -> {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(60.dp)
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = "모델 로딩 중...",
                        fontSize = 18.sp
                    )
                }
            }
            errorMessage != null -> {
                Text(
                    text = "오류: $errorMessage\n화면을 눌러 다시 시도하세요.",
                    textAlign = TextAlign.Center,
                    modifier = Modifier.padding(16.dp)
                )
            }
            selectedUri != null && isPlaying -> {
                // 실시간 비디오 처리 및 표시
                RealTimeVideoProcessing(
                    uri = selectedUri!!,
                    frameProcessor = frameProcessor,
                    onStop = {
                        isPlaying = false
                        selectedUri = null
                    },
                    onError = { error ->
                        errorMessage = error
                        isPlaying = false
                    }
                )
            }
            else -> {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center,
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(
                        text = "화면을 눌러서 비디오 선택",
                        fontSize = 20.sp,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "선택된 영상의 모든 프레임이 실시간으로 업스케일되어 표시됩니다",
                        textAlign = TextAlign.Center,
                        fontSize = 16.sp
                    )
                }
            }
        }
    }
}

@Composable
fun RealTimeVideoProcessing(
    uri: Uri,
    frameProcessor: FrameProcessor,
    onStop: () -> Unit,
    onError: (String) -> Unit
) {
    val context = LocalContext.current
    val processingActive = remember { AtomicBoolean(true) }
    val scope = rememberCoroutineScope()

    // 현재 프레임 상태
    val currentFrameState = remember { mutableStateOf<Bitmap?>(null) }
    val isFrameProcessing = remember { mutableStateOf(false) }

    // 비트맵 참조 관리를 위한 키
    val frameKey = remember { mutableStateOf(0) }

    // 비디오 처리 컴포넌트
    Box(modifier = Modifier.fillMaxSize()) {
        DisposableEffect(uri) {
            val job = scope.launch {
                try {
                    // 비디오 메타데이터 가져오기
                    val retriever = MediaMetadataRetriever()
                    retriever.setDataSource(context, uri)

                    val duration = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0
                    val fps = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)?.toFloatOrNull() ?: 30f
                    val frameInterval = (1000 / fps).toLong()

                    Log.d("VideoUpscale", "비디오 길이: ${duration}ms, FPS: $fps, 프레임 간격: ${frameInterval}ms")

                    // 비디오 처리 시작
                    var timeInMicros = 0L
                    while (isActive && processingActive.get() && timeInMicros <= duration * 1000) {
                        val frameTime = timeInMicros // 현재 타임스탬프에서 프레임 가져오기

                        // 이전 프레임 처리가 끝날 때까지 대기
                        if (isFrameProcessing.value) {
                            delay(5) // 작은 지연으로 CPU 점유율 감소
                            continue
                        }

                        isFrameProcessing.value = true

                        try {
                            // 다음 키 준비
                            val nextKey = frameKey.value + 1

                            withContext(Dispatchers.IO) {
                                // 프레임 가져오기
                                val originalFrame = retriever.getFrameAtTime(frameTime, MediaMetadataRetriever.OPTION_CLOSEST)

                                if (originalFrame != null) {
                                    try {
                                        // 프레임 처리 (복사본 생성 및 원본 해제)
                                        val processedFrame = frameProcessor.processFrame(originalFrame)

                                        // UI 스레드에서 상태 업데이트
                                        withContext(Dispatchers.Main) {
                                            // 이전 프레임 해제
                                            currentFrameState.value?.recycle()

                                            // 새 프레임 설정 및 키 업데이트
                                            currentFrameState.value = processedFrame
                                            frameKey.value = nextKey
                                        }
                                    } finally {
                                        // 원본 비트맵 항상 해제
                                        originalFrame.recycle()
                                    }
                                } else {
                                    Log.w("VideoUpscale", "프레임을 가져올 수 없음: $timeInMicros")
                                }
                            }

                            // 다음 프레임까지 딜레이
                            delay(frameInterval)

                        } catch (e: Exception) {
                            Log.e("VideoUpscale", "프레임 처리 중 오류", e)
                        } finally {
                            isFrameProcessing.value = false
                        }

                        // 다음 프레임 타임스탬프로 이동
                        timeInMicros += frameInterval * 1000
                    }

                    // 재생 완료
                    retriever.release()

                    withContext(Dispatchers.Main) {
                        // 마지막 프레임 해제
                        currentFrameState.value?.recycle()
                        currentFrameState.value = null
                        onStop()
                    }

                } catch (e: Exception) {
                    Log.e("VideoUpscale", "비디오 처리 중 오류 발생", e)

                    withContext(Dispatchers.Main) {
                        currentFrameState.value?.recycle()
                        currentFrameState.value = null
                        onError("비디오 처리 오류: ${e.localizedMessage}")
                    }
                }
            }

            onDispose {
                processingActive.set(false)
                job.cancel()

                // 최종 리소스 정리
                currentFrameState.value?.recycle()
                currentFrameState.value = null
            }
        }

        // 현재 프레임 표시 (키 변경시 재구성)
        val currentFrame = currentFrameState.value
        val key = frameKey.value

        if (currentFrame != null && !currentFrame.isRecycled) {
            key(key) {  // 키를 사용하여 이미지 재구성
                Image(
                    bitmap = currentFrame.asImageBitmap(),
                    contentDescription = "업스케일된 프레임",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Fit
                )
            }
        }

        // 안내 텍스트
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            contentAlignment = Alignment.BottomCenter
        ) {
            Surface(
                color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.7f),
                shape = MaterialTheme.shapes.medium
            ) {
                Text(
                    text = "영상 처리 중... 터치하여 정지",
                    modifier = Modifier
                        .padding(horizontal = 16.dp, vertical = 8.dp)
                        .clickable {
                            processingActive.set(false)
                            onStop()
                        },
                    fontSize = 16.sp
                )
            }
        }
    }
}

/**
 * 프레임 처리 클래스 - 비디오 프레임 업스케일링 처리
 */
class FrameProcessor {
    private var interpreter: Interpreter? = null
    private val lock = ReentrantLock()
    private var isModelLoaded = false

    suspend fun loadModel(context: android.content.Context) {
        if (isModelLoaded) return

        withContext(Dispatchers.IO) {
            lock.withLock {
                try {
                    val fileDescriptor = context.assets.openFd("ESPCN.tflite")
                    val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
                    val fileChannel = inputStream.channel
                    val startOffset = fileDescriptor.startOffset
                    val declaredLength = fileDescriptor.declaredLength
                    val model = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

                    // TFLite 인터프리터 옵션 설정
                    val options = Interpreter.Options().apply {
                        setNumThreads(4)  // 병렬 처리 활성화
                        addDelegate(FlexDelegate())  // 추가 연산자 지원
                    }

                    interpreter = Interpreter(model, options)
                    isModelLoaded = true

                    Log.d("FrameProcessor", "모델 로딩 완료")
                } catch (e: Exception) {
                    Log.e("FrameProcessor", "모델 로딩 실패", e)
                    throw e
                }
            }
        }
    }

    fun processFrame(frame: Bitmap): Bitmap {
        if (!isModelLoaded || interpreter == null) {
            Log.w("FrameProcessor", "모델이 로드되지 않았습니다")
            return Bitmap.createBitmap(frame) // 원본 비트맵의 복사본 반환
        }

        return lock.withLock {
            try {
                // 너무 큰 이미지는 효율성을 위해 스케일 다운
                // 크기 제한은 없지만 효율성을 위해 내부적으로 조정
                val scaleFactor = calculateOptimalScaleFactor(frame.width, frame.height)

                val scaledBitmap = if (scaleFactor < 1.0f) {
                    Bitmap.createScaledBitmap(
                        frame,
                        (frame.width * scaleFactor).toInt(),
                        (frame.height * scaleFactor).toInt(),
                        true
                    )
                } else {
                    // 복사본 생성 (원본에 영향 없도록)
                    Bitmap.createBitmap(frame)
                }

                // YCbCr 변환 및 입력 준비
                val (input, cr, cb) = extractYInputFromBitmap(scaledBitmap)

                // 출력 크기 계산 (3배 업스케일)
                val outputWidth = scaledBitmap.width * 3
                val outputHeight = scaledBitmap.height * 3

                // 출력 버퍼 준비
                val output = ByteBuffer.allocateDirect(outputWidth * outputHeight * 4)
                    .order(ByteOrder.nativeOrder())

                // 모델 입력 크기 조정
                val inputShape = intArrayOf(1, 1, scaledBitmap.height, scaledBitmap.width)
                interpreter?.resizeInput(0, inputShape)
                interpreter?.allocateTensors()

                // 모델 실행
                interpreter?.run(input, output)

                try {
                    // 결과 이미지 복원
                    val result = restoreBitmapFromY(output, cr, cb, outputHeight, outputWidth)
                    return result
                } finally {
                    // 중간 비트맵 해제 (원본과 다를 경우)
                    if (scaledBitmap !== frame) {
                        scaledBitmap.recycle()
                    }
                }
            } catch (e: Exception) {
                Log.e("FrameProcessor", "프레임 처리 실패", e)
                // 오류 시 원본의 복사본 반환
                return Bitmap.createBitmap(frame)
            }
        }
    }

    // 최적의 스케일 팩터 계산 (메모리/성능 최적화)
    private fun calculateOptimalScaleFactor(width: Int, height: Int): Float {
        val maxPixels = 1280 * 720 // HD 해상도 기준
        val imgPixels = width * height

        return if (imgPixels > maxPixels) {
            val factor = Math.sqrt(maxPixels.toDouble() / imgPixels)
            factor.toFloat()
        } else {
            1.0f // 축소 없음
        }
    }

    fun release() {
        lock.withLock {
            interpreter?.close()
            interpreter = null
            isModelLoaded = false
        }
    }
}

fun extractYInputFromBitmap(bitmap: Bitmap): Triple<ByteBuffer, ByteArray, ByteArray> {
    val width = bitmap.width
    val height = bitmap.height
    val inputSize = 1 * 1 * height * width  // NCHW 형식
    val inputBuffer = ByteBuffer.allocateDirect(inputSize * 4).order(ByteOrder.nativeOrder())
    val cb = ByteArray(width * height)
    val cr = ByteArray(width * height)

    val pixels = IntArray(width * height)
    bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

    for (i in pixels.indices) {
        val color = pixels[i]
        val r = (color shr 16) and 0xFF
        val g = (color shr 8) and 0xFF
        val b = color and 0xFF

        // YCbCr 표준 변환 공식 사용
        val y = (0.299 * r + 0.587 * g + 0.114 * b).toInt().coerceIn(0, 255)
        val cbVal = (128 + (-0.169 * r - 0.331 * g + 0.5 * b)).toInt().coerceIn(0, 255)
        val crVal = (128 + (0.5 * r - 0.419 * g - 0.081 * b)).toInt().coerceIn(0, 255)

        // 입력 정규화 (0~1)
        inputBuffer.putFloat(y / 255.0f)
        cb[i] = cbVal.toByte()
        cr[i] = crVal.toByte()
    }
    inputBuffer.rewind()
    return Triple(inputBuffer, cr, cb)
}

fun restoreBitmapFromY(
    yBuffer: ByteBuffer,
    cr: ByteArray,
    cb: ByteArray,
    height: Int,
    width: Int
): Bitmap {
    yBuffer.rewind()
    val output = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

    // 원본 이미지 크기
    val originalHeight = height / 3
    val originalWidth = width / 3

    for (y in 0 until height) {
        for (x in 0 until width) {
            val index = y * width + x

            if (index >= width * height) continue

            // 업스케일된 Y 값 가져오기 (0-1 범위에서 0-255로 변환)
            val yValue = (yBuffer.getFloat(index * 4) * 255).toInt().coerceIn(0, 255)

            // 원본 이미지의 좌표 계산 (다운샘플링)
            val originalY = (y * originalHeight / height).coerceIn(0, originalHeight - 1)
            val originalX = (x * originalWidth / width).coerceIn(0, originalWidth - 1)
            val originalIndex = originalY * originalWidth + originalX

            // 원본 이미지 사이즈 범위 확인
            if (originalIndex >= cb.size || originalIndex >= cr.size) continue

            // 바이트 값을 부호 없는 정수로 변환 (0-255)
            val crVal = cr[originalIndex].toInt() and 0xFF
            val cbVal = cb[originalIndex].toInt() and 0xFF

            // YCbCr을 RGB로 변환 (표준 BT.601 공식 사용)
            val r = (yValue + 1.402 * (crVal - 128)).toInt().coerceIn(0, 255)
            val g = (yValue - 0.344 * (cbVal - 128) - 0.714 * (crVal - 128)).toInt().coerceIn(0, 255)
            val b = (yValue + 1.772 * (cbVal - 128)).toInt().coerceIn(0, 255)

            output.setPixel(x, y, Color.rgb(r, g, b))
        }
    }
    return output
}