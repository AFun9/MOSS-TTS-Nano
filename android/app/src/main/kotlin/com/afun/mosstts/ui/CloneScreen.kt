package com.afun.mosstts.ui

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

@Composable
internal fun CloneScreen(state: TtsUiState, viewModel: TtsViewModel) {
    val busy = state.phase == Phase.Generating || state.phase == Phase.Initializing

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 16.dp, vertical = 12.dp),
    ) {
        StatusBar(state)
        Spacer(Modifier.height(12.dp))

        ReferenceAudioSection(
            state = state,
            viewModel = viewModel,
            busy = busy,
        )

        Spacer(Modifier.height(16.dp))

        OutlinedTextField(
            value = state.text,
            onValueChange = viewModel::onTextChanged,
            label = { Text("输入文本") },
            placeholder = { Text("输入要用克隆音色朗读的文字...") },
            modifier = Modifier.fillMaxWidth().weight(1f),
            enabled = !busy,
            maxLines = 12,
        )

        Spacer(Modifier.height(16.dp))

        ActionBar(
            state = state,
            onGenerate = viewModel::generateWithClone,
            onTogglePlay = viewModel::togglePlayback,
            generateEnabled = state.phase != Phase.Initializing
                    && state.text.isNotBlank()
                    && state.cloneAudioCodes != null,
        )
    }
}

// =====================================================================
// High-tech water-ripple orb driven by real-time mic amplitude.
// Layers (back → front):
//   1. Expanding sonar ripple rings (water-drop style)
//   2. Ambient glow halo
//   3. Rotating tech arcs (dashed orbit segments)
//   4. Glowing core sphere with inner gradient
//   5. Specular highlight
// =====================================================================

private val CyanPrimary = Color(0xFF00E5FF)
private val PurplePrimary = Color(0xFF7C4DFF)
private val BluePrimary = Color(0xFF448AFF)

@Composable
private fun RippleOrb(amplitude: Float, modifier: Modifier = Modifier) {
    val amp by animateFloatAsState(
        targetValue = amplitude,
        animationSpec = spring(dampingRatio = 0.55f, stiffness = 280f),
        label = "amp",
    )

    val inf = rememberInfiniteTransition(label = "orb")

    // 6 ripple rings staggered over a 3s cycle → one spawns every 500ms
    val ripplePhases = (0 until 6).map { i ->
        inf.animateFloat(
            initialValue = 0f, targetValue = 1f,
            animationSpec = infiniteRepeatable(
                tween(3000, easing = LinearEasing),
                RepeatMode.Restart,
                initialStartOffset = androidx.compose.animation.core.StartOffset(i * 500),
            ),
            label = "rip$i",
        )
    }

    // Slow rotation for the tech-arc orbits
    val rot1 by inf.animateFloat(
        0f, 360f,
        infiniteRepeatable(tween(8000, easing = LinearEasing), RepeatMode.Restart),
        label = "rot1",
    )
    val rot2 by inf.animateFloat(
        360f, 0f,
        infiniteRepeatable(tween(12000, easing = LinearEasing), RepeatMode.Restart),
        label = "rot2",
    )
    val rot3 by inf.animateFloat(
        0f, 360f,
        infiniteRepeatable(tween(6000, easing = LinearEasing), RepeatMode.Restart),
        label = "rot3",
    )

    // Breathing pulse for the core
    val breath by inf.animateFloat(
        0f, (2 * PI).toFloat(),
        infiniteRepeatable(tween(2400, easing = LinearEasing), RepeatMode.Restart),
        label = "breath",
    )

    Canvas(
        modifier = modifier.background(Color.Transparent),
    ) {
        val cx = size.width / 2f
        val cy = size.height / 2f
        val half = min(size.width, size.height) / 2f
        val center = Offset(cx, cy)
        val aFactor = 0.25f + amp * 0.75f

        // --- Layer 1: Expanding sonar ripple rings ---
        for (phaseState in ripplePhases) {
            val t = phaseState.value
            val radius = half * (0.25f + 0.75f * t) * (0.85f + 0.15f * aFactor)
            val alpha = (1f - t) * 0.45f * aFactor
            if (alpha > 0.01f) {
                drawCircle(
                    color = CyanPrimary.copy(alpha = alpha),
                    radius = radius,
                    center = center,
                    style = Stroke(width = (2.5f - 1.5f * t).coerceAtLeast(0.5f).dp.toPx()),
                )
            }
        }

        // --- Layer 2: Ambient glow halo ---
        val glowR = half * 0.55f * (0.9f + 0.2f * sin(breath))
        drawCircle(
            brush = Brush.radialGradient(
                colors = listOf(
                    PurplePrimary.copy(alpha = 0.18f * aFactor),
                    BluePrimary.copy(alpha = 0.06f * aFactor),
                    Color.Transparent,
                ),
                center = center,
                radius = glowR * 1.6f,
            ),
            radius = glowR * 1.6f,
            center = center,
        )

        // --- Layer 3: Rotating tech arcs ---
        val arcStroke = Stroke(
            width = 1.8f.dp.toPx(),
            cap = StrokeCap.Round,
        )
        fun drawTechArc(rotation: Float, radius: Float, sweep: Float, color: Color) {
            val d = radius * 2
            val topLeft = Offset(cx - radius, cy - radius)
            rotate(rotation, pivot = center) {
                drawArc(
                    color = color,
                    startAngle = 0f,
                    sweepAngle = sweep,
                    useCenter = false,
                    topLeft = topLeft,
                    size = Size(d, d),
                    style = arcStroke,
                )
                drawArc(
                    color = color,
                    startAngle = 180f,
                    sweepAngle = sweep * 0.6f,
                    useCenter = false,
                    topLeft = topLeft,
                    size = Size(d, d),
                    style = arcStroke,
                )
            }
        }
        val arcR1 = half * (0.52f + 0.06f * amp)
        val arcR2 = half * (0.42f + 0.04f * amp)
        val arcR3 = half * (0.62f + 0.08f * amp)
        drawTechArc(rot1, arcR1, 70f + 30f * amp, CyanPrimary.copy(alpha = 0.6f))
        drawTechArc(rot2, arcR2, 50f + 20f * amp, PurplePrimary.copy(alpha = 0.45f))
        drawTechArc(rot3, arcR3, 40f + 35f * amp, BluePrimary.copy(alpha = 0.35f))

        // --- Layer 4: Core sphere ---
        val breathScale = 0.92f + 0.08f * sin(breath)
        val coreR = half * 0.22f * (0.7f + 0.6f * amp) * breathScale

        // Outer glow of core
        drawCircle(
            brush = Brush.radialGradient(
                colors = listOf(
                    CyanPrimary.copy(alpha = 0.4f * aFactor),
                    PurplePrimary.copy(alpha = 0.08f),
                    Color.Transparent,
                ),
                center = center,
                radius = coreR * 2.2f,
            ),
            radius = coreR * 2.2f,
            center = center,
        )

        // Solid core
        drawCircle(
            brush = Brush.radialGradient(
                colors = listOf(
                    Color.White.copy(alpha = 0.9f),
                    CyanPrimary,
                    PurplePrimary,
                    BluePrimary.copy(alpha = 0.6f),
                ),
                center = Offset(cx - coreR * 0.15f, cy - coreR * 0.18f),
                radius = coreR * 1.1f,
            ),
            radius = coreR,
            center = center,
        )

        // --- Layer 5: Specular highlight ---
        val specR = coreR * 0.3f
        drawCircle(
            brush = Brush.radialGradient(
                colors = listOf(
                    Color.White.copy(alpha = 0.7f),
                    Color.White.copy(alpha = 0f),
                ),
                center = Offset(cx - coreR * 0.22f, cy - coreR * 0.25f),
                radius = specR,
            ),
            radius = specR,
            center = Offset(cx - coreR * 0.22f, cy - coreR * 0.25f),
        )

        // Small secondary dot particles on the orbit
        val dotCount = 8
        for (i in 0 until dotCount) {
            val angle = (rot1 / 180f * PI).toFloat() + i * (2f * PI.toFloat() / dotCount)
            val orbitR = half * (0.48f + 0.04f * sin(breath + i))
            val dx = cx + cos(angle) * orbitR
            val dy = cy + sin(angle) * orbitR
            val dotAlpha = (0.3f + 0.4f * amp) * (0.5f + 0.5f * sin(breath + i * 0.7f))
            drawCircle(
                color = CyanPrimary.copy(alpha = dotAlpha),
                radius = 2.dp.toPx(),
                center = Offset(dx, dy),
            )
        }
    }
}

// =====================================================================
// Reference Audio Section
// =====================================================================

@Composable
private fun ReferenceAudioSection(
    state: TtsUiState,
    viewModel: TtsViewModel,
    busy: Boolean,
) {
    val context = LocalContext.current

    val filePicker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri -> uri?.let { viewModel.importAudio(it) } }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) viewModel.startRecording() }

    Surface(
        shape = RoundedCornerShape(16.dp),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                "参考音频",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Medium,
            )
            Spacer(Modifier.height(12.dp))

            when {
                // ---- Recording: show ripple orb ----
                state.isRecording -> {
                    val amp by viewModel.recordingAmplitude.collectAsState()
                    val secs by viewModel.recordingSeconds.collectAsState()

                    RippleOrb(
                        amplitude = amp,
                        modifier = Modifier.size(160.dp),
                    )
                    Spacer(Modifier.height(8.dp))
                    Text(
                        "录音中  %.1fs / %ds".format(secs, 20),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Spacer(Modifier.height(12.dp))
                    Button(
                        onClick = viewModel::stopRecording,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.error,
                        ),
                    ) {
                        Icon(Icons.Default.Stop, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(6.dp))
                        Text("停止录音")
                    }
                }

                // ---- Encoding ----
                state.isEncoding -> {
                    CircularProgressIndicator(
                        modifier = Modifier.size(48.dp),
                        strokeWidth = 3.dp,
                    )
                    Spacer(Modifier.height(12.dp))
                    Text(
                        "正在编码参考音频...",
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }

                // ---- Ready ----
                state.cloneAudioCodes != null -> {
                    Icon(
                        Icons.Default.CheckCircle,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(40.dp),
                    )
                    Spacer(Modifier.height(8.dp))
                    Text(
                        "参考音频已就绪",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Medium,
                    )
                    Text(
                        "%.1f 秒 · %d 帧".format(
                            state.cloneAudioDuration,
                            state.cloneAudioCodes.frames,
                        ),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Spacer(Modifier.height(12.dp))
                    OutlinedButton(
                        onClick = viewModel::clearCloneAudio,
                        enabled = !busy,
                    ) {
                        Icon(Icons.Default.Close, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(Modifier.width(4.dp))
                        Text("清除，重新选择")
                    }
                }

                // ---- Empty: record + import ----
                else -> {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                    ) {
                        Button(
                            onClick = {
                                val hasPerm = ContextCompat.checkSelfPermission(
                                    context, Manifest.permission.RECORD_AUDIO
                                ) == PackageManager.PERMISSION_GRANTED
                                if (hasPerm) viewModel.startRecording()
                                else permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                            },
                            enabled = !busy,
                            modifier = Modifier.weight(1f).height(56.dp),
                        ) {
                            Icon(Icons.Default.Mic, contentDescription = null)
                            Spacer(Modifier.width(8.dp))
                            Text("录音")
                        }
                        OutlinedButton(
                            onClick = { filePicker.launch(arrayOf("audio/*")) },
                            enabled = !busy,
                            modifier = Modifier.weight(1f).height(56.dp),
                        ) {
                            Icon(Icons.Default.FolderOpen, contentDescription = null)
                            Spacer(Modifier.width(8.dp))
                            Text("导入文件")
                        }
                    }
                    Spacer(Modifier.height(8.dp))
                    Text(
                        "提供 3~20 秒的参考音频，用于克隆音色",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
            }
        }
    }
}
