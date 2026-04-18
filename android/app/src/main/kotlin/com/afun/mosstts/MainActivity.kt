package com.afun.mosstts

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.lifecycle.viewmodel.compose.viewModel
import com.afun.mosstts.ui.MainScreen
import com.afun.mosstts.ui.TtsViewModel
import com.afun.mosstts.ui.theme.MossTtsTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MossTtsTheme {
                val vm: TtsViewModel = viewModel()
                MainScreen(vm)
            }
        }
    }
}
