package com.afun.mosstts

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.afun.mosstts.ui.theme.MossTtsTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MossTtsTheme {
                Surface { HelloScreen() }
            }
        }
    }
}

@Composable
private fun HelloScreen() {
    Box(modifier = Modifier.fillMaxSize().padding(24.dp)) {
        Text("MOSS TTS Nano · v0.1.0")
    }
}
