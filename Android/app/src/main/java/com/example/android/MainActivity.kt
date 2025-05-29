package com.example.android

import android.content.pm.ActivityInfo
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.util.Log
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
import androidx.compose.animation.*
import androidx.compose.ui.draw.scale
import com.example.android.ui.theme.AndroidTheme
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil
import kotlin.math.min
import androidx.core.graphics.createBitmap
import androidx.compose.animation.core.tween

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 가로 모드로 고정
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE

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

    var upscaleEnabled by remember { mutableStateOf(true) }

    val frameProcessor = remember { FrameProcessor() }

    LaunchedEffect(Unit) {
        isLoading = true
        try {
            frameProcessor.loadModel(context)
        } catch (e: Exception) {
            Log.e("VideoUpscale", "모델 로딩 실패", e)
            errorMessage = "모델 로딩 실패"
        } finally {
            isLoading = false
        }
    }

    val getVideoUri = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        selectedUri = uri
        errorMessage = null
        if (uri != null) {
            isPlaying = true
        }
    }

    DisposableEffect(Unit) {
        onDispose {
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
                    CircularProgressIndicator()
                    Spacer(modifier = Modifier.height(16.dp))
                    Text("모델 로딩 중...")
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
                RealTimeVideoProcessing(
                    uri = selectedUri!!,
                    frameProcessor = frameProcessor,
                    upscaleEnabled = upscaleEnabled,
                    onUpscaleToggle = { upscaleEnabled = it },
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
                Text(
                    text = "화면을 눌러서 비디오 선택",
                    fontSize = 20.sp,
                    textAlign = TextAlign.Center
                )
            }
        }

        // 화면 하단 업스케일 ON/OFF 토글 (비디오 재생 중이 아닐 때만 표시)
        if (!isPlaying) {
            Column(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 36.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("업스케일")
                    Switch(
                        checked = upscaleEnabled,
                        onCheckedChange = { upscaleEnabled = it }
                    )
                    Text(if (upscaleEnabled) "ON" else "OFF")
                }
            }
        }
    }
}

@Composable
fun RealTimeVideoProcessing(
    uri: Uri,
    frameProcessor: FrameProcessor,
    upscaleEnabled: Boolean,
    onUpscaleToggle: (Boolean) -> Unit,
    onStop: () -> Unit,
    onError: (String) -> Unit
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val isActive = remember { mutableStateOf(true) }
    val currentFrameState = remember { mutableStateOf<Bitmap?>(null) }
    val frameKey = remember { mutableIntStateOf(0) }

    // 비디오 재생 제어 상태
    var isPlayingState by remember { mutableStateOf(true) }
    var currentPosition by remember { mutableStateOf(0L) }
    var totalDuration by remember { mutableStateOf(0L) }
    var seekPosition by remember { mutableStateOf<Long?>(null) }

    // UI 표시/숨김 상태
    var showControls by remember { mutableStateOf(true) }
    var hideControlsJob: Job? by remember { mutableStateOf(null) }

    val upscaleEnabledState = rememberUpdatedState(upscaleEnabled)

    // UI 자동 숨김 함수
    fun scheduleHideControls() {
        hideControlsJob?.cancel()
        hideControlsJob = scope.launch {
            delay(3000) // 3초 후 숨김
            showControls = false
        }
    }

    // UI 표시 함수
    fun showControlsAndScheduleHide() {
        hideControlsJob?.cancel()
        showControls = true
        if (isPlayingState) {
            scheduleHideControls()
        }
    }

    // UI 토글 함수 (터치로 켜고 끄기)
    fun toggleControls() {
        hideControlsJob?.cancel()
        showControls = !showControls
        if (showControls && isPlayingState) {
            scheduleHideControls()
        }
    }

    // 재생 상태가 변경될 때 UI 숨김 스케줄링
    LaunchedEffect(isPlayingState) {
        if (isPlayingState && showControls) {
            scheduleHideControls()
        } else if (!isPlayingState) {
            hideControlsJob?.cancel()
        }
    }

    // 컴포넌트 시작 시 UI 숨김 스케줄링
    LaunchedEffect(Unit) {
        scheduleHideControls()
    }

    Box(modifier = Modifier.fillMaxSize()) {
        DisposableEffect(uri) {
            val job = scope.launch {
                try {
                    val retriever = MediaMetadataRetriever()
                    retriever.setDataSource(context, uri)

                    val duration = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0
                    totalDuration = duration
                    val fps = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)?.toFloatOrNull() ?: 30f
                    val frameInterval = (1000 / fps).toLong()

                    var timeInMicros = 0L

                    while (isActive.value && timeInMicros <= duration * 1000) {
                        // Seek 요청이 있는 경우 처리
                        seekPosition?.let { seekPos ->
                            timeInMicros = seekPos * 1000
                            seekPosition = null
                        }

                        // 일시정지 상태인 경우 대기
                        while (!isPlayingState && isActive.value) {
                            delay(100)
                        }

                        if (!isActive.value) break

                        withContext(Dispatchers.IO) {
                            val originalFrame = retriever.getFrameAtTime(timeInMicros, MediaMetadataRetriever.OPTION_CLOSEST)
                            if (originalFrame != null) {
                                val processedFrame = if (upscaleEnabledState.value) {
                                    frameProcessor.processFrame(originalFrame)
                                } else {
                                    frameProcessor.bilinearScale(
                                        originalFrame,
                                        originalFrame.width * 3,
                                        originalFrame.height * 3
                                    )
                                }
                                originalFrame.recycle()
                                withContext(Dispatchers.Main) {
                                    currentFrameState.value?.recycle()
                                    currentFrameState.value = processedFrame
                                    frameKey.intValue += 1
                                    currentPosition = timeInMicros / 1000
                                }
                            }
                        }

                        delay(frameInterval)
                        timeInMicros += frameInterval * 1000
                    }

                    retriever.release()
                    withContext(Dispatchers.Main) {
                        currentFrameState.value?.recycle()
                        currentFrameState.value = null
                        onStop()
                    }
                } catch (e: Exception) {
                    Log.e("VideoUpscale", "비디오 처리 오류", e)
                    withContext(Dispatchers.Main) {
                        currentFrameState.value?.recycle()
                        currentFrameState.value = null
                        onError("비디오 처리 오류")
                    }
                }
            }
            onDispose {
                isActive.value = false
                job.cancel()
                hideControlsJob?.cancel()
                currentFrameState.value?.recycle()
                currentFrameState.value = null
            }
        }

        val currentFrame = currentFrameState.value
        val key = frameKey.intValue

        // 비디오 프레임 (전체 화면에 클릭 가능)
        if (currentFrame != null && !currentFrame.isRecycled) {
            key(key) {
                Image(
                    bitmap = currentFrame.asImageBitmap(),
                    contentDescription = "업스케일된 프레임",
                    modifier = Modifier
                        .fillMaxSize()
                        .clickable(
                            interactionSource = remember { androidx.compose.foundation.interaction.MutableInteractionSource() },
                            indication = null
                        ) {
                            toggleControls() // 터치로 UI 토글
                        },
                    contentScale = ContentScale.Fit
                )
            }
        }

        // 비디오 컨트롤 UI (애니메이션과 함께 표시/숨김)
        AnimatedVisibility(
            visible = showControls,
            enter = slideInVertically(
                initialOffsetY = { it },
                animationSpec = tween(300)
            ) + fadeIn(animationSpec = tween(300)),
            exit = slideOutVertically(
                targetOffsetY = { it },
                animationSpec = tween(300)
            ) + fadeOut(animationSpec = tween(300)),
            modifier = Modifier.align(Alignment.BottomCenter)
        ) {
            VideoControls(
                isPlaying = isPlayingState,
                currentPosition = currentPosition,
                totalDuration = totalDuration,
                onPlayPause = {
                    isPlayingState = !isPlayingState
                    showControlsAndScheduleHide()
                },
                onSeek = { position ->
                    seekPosition = position
                    showControlsAndScheduleHide()
                },
                onStop = onStop,
                upscaleEnabled = upscaleEnabled,
                onUpscaleToggle = { enabled ->
                    onUpscaleToggle(enabled)
                    showControlsAndScheduleHide()
                },
                modifier = Modifier
            )
        }
    }
}

@Composable
fun VideoControls(
    isPlaying: Boolean,
    currentPosition: Long,
    totalDuration: Long,
    onPlayPause: () -> Unit,
    onSeek: (Long) -> Unit,
    onStop: () -> Unit,
    upscaleEnabled: Boolean,
    onUpscaleToggle: (Boolean) -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(16.dp)
    ) {
        // 슬라이더 위치 상태
        var sliderPosition by remember { mutableStateOf(
            if (totalDuration > 0) (currentPosition.toFloat() / totalDuration.toFloat()).coerceIn(0f, 1f)
            else 0f
        ) }

        // 프로그레스 바
        if (totalDuration > 0) {
            // 현재 위치 변할 때마다 sliderPosition도 연동
            LaunchedEffect(currentPosition, totalDuration) {
                sliderPosition = if (totalDuration > 0)
                    (currentPosition.toFloat() / totalDuration.toFloat()).coerceIn(0f, 1f)
                else 0f
            }

            Column {
                // 시간 표시
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = formatTime(currentPosition),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    Text(
                        text = formatTime(totalDuration),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }

                // 프로그레스 바 (트랙 클릭, 드래그 모두 Seek)
                Slider(
                    value = sliderPosition,
                    onValueChange = { newProgress ->
                        sliderPosition = newProgress
                    },
                    onValueChangeFinished = {
                        val newPosition = (sliderPosition * totalDuration).toLong()
                        onSeek(newPosition)
                    },
                    modifier = Modifier.fillMaxWidth()
                )
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        // 컨트롤 버튼들
        Surface(
            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.9f),
            shape = MaterialTheme.shapes.medium
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(16.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // 재생/일시정지 버튼
                Button(
                    onClick = onPlayPause,
                    modifier = Modifier.width(60.dp).height(48.dp)
                ) {
                    Text(
                        text = if (isPlaying) "⏸" else "▶",
                        fontSize = 12.sp
                    )
                }

                // 정지 버튼
                Button(
                    onClick = onStop,
                    modifier = Modifier.width(60.dp).height(48.dp)
                ) {
                    Text(
                        text = "⏹",
                        fontSize = 12.sp
                    )
                }

                Column {
                    Text(
                        text = if (isPlaying) "재생 중" else "일시정지",
                        style = MaterialTheme.typography.bodyMedium
                    )

                    // 업스케일 토글
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Text(
                            text = "업스케일",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                        )
                        Switch(
                            checked = upscaleEnabled,
                            onCheckedChange = onUpscaleToggle,
                            modifier = Modifier.scale(0.8f)
                        )
                        Text(
                            text = if (upscaleEnabled) "ON" else "OFF",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                        )
                    }
                }
            }
        }
    }
}

fun formatTime(timeMs: Long): String {
    val totalSeconds = timeMs / 1000
    val minutes = totalSeconds / 60
    val seconds = totalSeconds % 60
    return "%02d:%02d".format(minutes, seconds)
}

class FrameProcessor {
    private var interpreter: Interpreter? = null
    private var nnApiDelegate: NnApiDelegate? = null
    private var isModelLoaded = false

    private val PATCH_SIZE = 128

    fun loadModel(context: android.content.Context) {
        if (isModelLoaded) return

        try {
            val nnApiOptions = NnApiDelegate.Options().apply {
                allowFp16 = true
                useNnapiCpu = true
                executionPreference = NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED
            }
            nnApiDelegate = NnApiDelegate(nnApiOptions)
            val options = Interpreter.Options().apply {
                numThreads = 8
                addDelegate(nnApiDelegate)
            }
            val modelBytes = context.assets.open("XLSR_W8A8.tflite").readBytes()
            val modelBuffer = ByteBuffer.allocateDirect(modelBytes.size).apply {
                order(ByteOrder.nativeOrder())
                put(modelBytes)
                rewind()
            }
            interpreter = Interpreter(modelBuffer, options)
            isModelLoaded = true
            Log.d("FrameProcessor", "NPU 모델 로딩 성공")
        } catch (e: Exception) {
            Log.w("FrameProcessor", "NPU 모드 로딩 실패, CPU 모드로 시도", e)
            try {
                nnApiDelegate?.close()
                nnApiDelegate = null
                interpreter?.close()
                val options = Interpreter.Options().apply { numThreads = 4 }
                val modelBytes = context.assets.open("XLSR_W8A8.tflite").readBytes()
                val modelBuffer = ByteBuffer.allocateDirect(modelBytes.size).apply {
                    order(ByteOrder.nativeOrder())
                    put(modelBytes)
                    rewind()
                }
                interpreter = Interpreter(modelBuffer, options)
                isModelLoaded = true
                Log.d("FrameProcessor", "CPU 모델 로딩 성공")
            } catch (e2: Exception) {
                Log.e("FrameProcessor", "모델 로딩 최종 실패", e2)
                throw e2
            }
        }
    }

    fun processFrame(frame: Bitmap): Bitmap {
        if (!isModelLoaded || interpreter == null) {
            return bilinearScale(frame, frame.width * 3, frame.height * 3)
        }
        val patchRowCount = ceil(frame.height / PATCH_SIZE.toFloat()).toInt()
        val patchColCount = ceil(frame.width / PATCH_SIZE.toFloat()).toInt()

        var patchOutputWidth = 0
        var patchOutputHeight = 0

        run {
            val tempPatch = Bitmap.createBitmap(
                frame, 0, 0,
                min(PATCH_SIZE, frame.width),
                min(PATCH_SIZE, frame.height)
            )
            val interpreter = this.interpreter!!
            val outputTensor = interpreter.getOutputTensor(0)
            val outShape = outputTensor.shape()
            patchOutputHeight = outShape[1]
            patchOutputWidth = outShape[2]
            tempPatch.recycle()
        }

        val outputWidth = (frame.width * patchOutputWidth) / PATCH_SIZE
        val outputHeight = (frame.height * patchOutputHeight) / PATCH_SIZE

        val resultBitmap = createBitmap(outputWidth, outputHeight)
        val canvas = Canvas(resultBitmap)
        canvas.drawColor(Color.BLACK)

        for (row in 0 until patchRowCount) {
            for (col in 0 until patchColCount) {
                val startX = col * PATCH_SIZE
                val startY = row * PATCH_SIZE
                val endX = min(startX + PATCH_SIZE, frame.width)
                val endY = min(startY + PATCH_SIZE, frame.height)
                val patchWidth = endX - startX
                val patchHeight = endY - startY

                val patch = if (patchWidth == PATCH_SIZE && patchHeight == PATCH_SIZE) {
                    Bitmap.createBitmap(frame, startX, startY, PATCH_SIZE, PATCH_SIZE)
                } else {
                    val paddedPatch = createBitmap(PATCH_SIZE, PATCH_SIZE)
                    val tempCanvas = Canvas(paddedPatch)
                    tempCanvas.drawColor(Color.BLACK)
                    tempCanvas.drawBitmap(
                        frame,
                        Rect(startX, startY, endX, endY),
                        Rect(0, 0, patchWidth, patchHeight),
                        null
                    )
                    paddedPatch
                }
                val upscaledPatch = processPatchDynamic(patch)
                patch.recycle()

                val drawWidth = (patchWidth * patchOutputWidth) / PATCH_SIZE
                val drawHeight = (patchHeight * patchOutputHeight) / PATCH_SIZE

                val croppedUpscaled = if (drawWidth != upscaledPatch.width || drawHeight != upscaledPatch.height) {
                    Bitmap.createBitmap(upscaledPatch, 0, 0, drawWidth, drawHeight)
                } else {
                    upscaledPatch
                }

                canvas.drawBitmap(
                    croppedUpscaled,
                    null,
                    Rect(
                        col * patchOutputWidth,
                        row * patchOutputHeight,
                        col * patchOutputWidth + drawWidth,
                        row * patchOutputHeight + drawHeight
                    ),
                    null
                )
                if (croppedUpscaled != upscaledPatch) croppedUpscaled.recycle()
                upscaledPatch.recycle()
            }
        }
        return resultBitmap
    }

    private fun processPatchDynamic(patch: Bitmap): Bitmap {
        val interpreter = this.interpreter ?: return bilinearScale(patch, patch.width * 3, patch.height * 3)
        val inputBuffer = prepareModelInputNHWC(patch)
        val outputTensor = interpreter.getOutputTensor(0)
        val outShape = outputTensor.shape()
        val outHeight = outShape[1]
        val outWidth = outShape[2]
        val outChannels = outShape[3]
        val outputBuffer = ByteBuffer.allocateDirect(outHeight * outWidth * outChannels)
            .order(ByteOrder.nativeOrder())

        try {
            interpreter.run(inputBuffer, outputBuffer)
            return reconstructImageNHWC(outputBuffer, outWidth, outHeight)
        } catch (e: Exception) {
            Log.e("FrameProcessor", "모델 실행 오류: ${e.message}")
            e.printStackTrace()
            return bilinearScale(patch, patch.width * 3, patch.height * 3)
        }
    }

    private fun prepareModelInputNHWC(bitmap: Bitmap): ByteBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        val inputBuffer = ByteBuffer.allocateDirect(1 * height * width * 3).order(ByteOrder.nativeOrder())
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = pixels[y * width + x]
                inputBuffer.put(((pixel shr 16) and 0xFF).toByte())
                inputBuffer.put(((pixel shr 8) and 0xFF).toByte())
                inputBuffer.put((pixel and 0xFF).toByte())
            }
        }
        inputBuffer.rewind()
        return inputBuffer
    }

    private fun reconstructImageNHWC(
        outputBuffer: ByteBuffer,
        width: Int,
        height: Int
    ): Bitmap {
        outputBuffer.rewind()
        val output = createBitmap(width, height)
        val pixels = IntArray(width * height)
        for (i in 0 until width * height) {
            val r = outputBuffer.get().toInt() and 0xFF
            val g = outputBuffer.get().toInt() and 0xFF
            val b = outputBuffer.get().toInt() and 0xFF
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        output.setPixels(pixels, 0, width, 0, 0, width, height)
        return output
    }

    fun bilinearScale(src: Bitmap, destWidth: Int, destHeight: Int): Bitmap {
        val result = createBitmap(destWidth, destHeight)
        val canvas = Canvas(result)
        canvas.drawBitmap(
            src,
            Rect(0, 0, src.width, src.height),
            Rect(0, 0, destWidth, destHeight),
            null
        )
        return result
    }

    fun release() {
        interpreter?.close()
        interpreter = null
        nnApiDelegate?.close()
        nnApiDelegate = null
        isModelLoaded = false
    }
}