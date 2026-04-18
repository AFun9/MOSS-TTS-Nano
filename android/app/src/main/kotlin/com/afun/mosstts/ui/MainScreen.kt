package com.afun.mosstts.ui

import androidx.compose.animation.animateColorAsState
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.afun.mosstts.core.voice.BuiltinVoice

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(viewModel: TtsViewModel) {
    val state by viewModel.state.collectAsState()
    val snackbar = remember { SnackbarHostState() }

    LaunchedEffect(state.errorMessage) {
        state.errorMessage?.let {
            snackbar.showSnackbar(it)
            viewModel.dismissError()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "MOSS TTS Nano",
                        fontWeight = FontWeight.Bold,
                    )
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer,
                ),
            )
        },
        snackbarHost = { SnackbarHost(snackbar) },
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 16.dp, vertical = 12.dp),
        ) {
            // --- Status chip ---
            StatusBar(state)

            Spacer(Modifier.height(12.dp))

            // --- Text input ---
            OutlinedTextField(
                value = state.text,
                onValueChange = viewModel::onTextChanged,
                label = { Text("输入文本") },
                placeholder = { Text("在这里输入要朗读的文字...") },
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                enabled = state.phase != Phase.Generating && state.phase != Phase.Initializing,
                maxLines = 12,
            )

            Spacer(Modifier.height(12.dp))

            // --- Voice selector ---
            Text(
                "选择音色",
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(6.dp))
            VoicePicker(
                voices = state.voices,
                selectedIdx = state.selectedVoiceIdx,
                onSelect = viewModel::onVoiceSelected,
                enabled = state.phase != Phase.Generating && state.phase != Phase.Initializing,
            )

            Spacer(Modifier.height(16.dp))

            // --- Action bar ---
            ActionBar(
                state = state,
                onGenerate = viewModel::generate,
                onTogglePlay = viewModel::togglePlayback,
            )
        }
    }
}

@Composable
private fun StatusBar(state: TtsUiState) {
    val isStreaming = state.phase == Phase.Generating && state.isPlaying
    val text = when {
        state.phase == Phase.Idle -> "就绪"
        state.phase == Phase.Initializing -> "正在加载模型..."
        isStreaming -> "流式播放中 · 已生成 ${state.framesGenerated} 帧"
        state.phase == Phase.Generating && state.framesGenerated > 0 ->
            "正在生成 · ${state.framesGenerated} 帧"
        state.phase == Phase.Generating -> "正在生成语音..."
        state.phase == Phase.Ready && state.isPlaying -> "播放中..."
        state.phase == Phase.Ready ->
            "完毕 · ${state.generationTimeMs}ms · %.1fs".format(state.audioDurationSec)
        state.phase == Phase.Error -> "出错了"
        else -> ""
    }
    val color by animateColorAsState(
        when (state.phase) {
            Phase.Error -> MaterialTheme.colorScheme.error
            Phase.Ready -> MaterialTheme.colorScheme.primary
            else -> MaterialTheme.colorScheme.onSurfaceVariant
        },
        label = "statusColor",
    )
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.fillMaxWidth(),
    ) {
        if (state.phase == Phase.Initializing || state.phase == Phase.Generating) {
            CircularProgressIndicator(
                modifier = Modifier.size(16.dp),
                strokeWidth = 2.dp,
            )
            Spacer(Modifier.width(8.dp))
        }
        Text(text, style = MaterialTheme.typography.bodyMedium, color = color)
    }
}

@Composable
private fun VoicePicker(
    voices: List<BuiltinVoice>,
    selectedIdx: Int,
    onSelect: (Int) -> Unit,
    enabled: Boolean,
) {
    val scrollState = rememberScrollState()
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .horizontalScroll(scrollState),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        voices.forEachIndexed { idx, voice ->
            val selected = idx == selectedIdx
            val containerColor by animateColorAsState(
                if (selected) MaterialTheme.colorScheme.primaryContainer
                else MaterialTheme.colorScheme.surface,
                label = "voiceChip$idx",
            )
            Surface(
                onClick = { if (enabled) onSelect(idx) },
                shape = RoundedCornerShape(20.dp),
                color = containerColor,
                border = BorderStroke(
                    1.dp,
                    if (selected) MaterialTheme.colorScheme.primary
                    else MaterialTheme.colorScheme.outlineVariant,
                ),
            ) {
                Column(
                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 8.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Text(
                        voice.voice,
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
                        color = if (selected) MaterialTheme.colorScheme.primary
                                else MaterialTheme.colorScheme.onSurface,
                    )
                    Text(
                        voice.group,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

@Composable
private fun ActionBar(
    state: TtsUiState,
    onGenerate: () -> Unit,
    onTogglePlay: () -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Button(
            onClick = onGenerate,
            enabled = state.phase != Phase.Generating
                    && state.phase != Phase.Initializing
                    && state.text.isNotBlank(),
            modifier = Modifier.weight(1f),
            contentPadding = PaddingValues(vertical = 14.dp),
        ) {
            if (state.phase == Phase.Generating) {
                CircularProgressIndicator(
                    modifier = Modifier.size(18.dp),
                    strokeWidth = 2.dp,
                    color = MaterialTheme.colorScheme.onPrimary,
                )
                Spacer(Modifier.width(8.dp))
                if (state.framesGenerated > 0) {
                    Text("${state.framesGenerated} 帧")
                } else {
                    Text("生成中...")
                }
            } else {
                Text("生成语音")
            }
        }

        FilledTonalButton(
            onClick = onTogglePlay,
            enabled = state.phase == Phase.Ready || (state.phase == Phase.Generating && state.isPlaying),
            modifier = Modifier.weight(1f),
            contentPadding = PaddingValues(vertical = 14.dp),
        ) {
            Text(if (state.isPlaying) "停止" else "播放")
        }
    }
}
