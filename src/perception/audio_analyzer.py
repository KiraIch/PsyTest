"""音訊特徵分析（Phase 5 實作，純 numpy，無重型依賴）。

輸入音訊波形（建議 16kHz，任意取樣率皆可），輸出結構化語音分析結果。
刻意僅依賴 numpy：librosa/parselmouth 在部分平台安裝困難，
且管線只需穩定的基礎特徵。所有欄位皆為真實計算，
``speech_rate`` 為基於能量起伏的近似估計（已於欄位註明）。

使用範例：
    from src.perception.audio_analyzer import AudioAnalyzer

    features = AudioAnalyzer(sample_rate=24000).analyze(waveform)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class AudioFeatures:
    """音訊特徵結構。"""

    pitch_mean: float = 0.0
    pitch_variance: float = 0.0
    pitch_contour: List[float] = field(default_factory=list)
    energy: float = 0.0
    speech_rate: float = 0.0  # 近似估計（能量起伏法），非精確音節切分
    pause_segments: List[Tuple[float, float]] = field(default_factory=list)
    jitter: float = 0.0
    shimmer: float = 0.0
    spectral_features: Dict[str, float] = field(default_factory=dict)
    duration: float = 0.0


def _frame_energy(waveform: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """計算每幀 RMS 能量。"""
    if waveform.size < frame_len:
        return np.array([float(np.sqrt(np.mean(waveform**2)))])
    frames = np.lib.stride_tricks.sliding_window_view(waveform, frame_len)[::hop]
    return np.sqrt(np.mean(frames**2, axis=1))


def _pitch_autocorr(
    voiced: np.ndarray, sample_rate: int, fmin: float = 50.0, fmax: float = 500.0
) -> List[float]:
    """以自相關法估計基頻軌跡（每 30ms 一點，無聲段以 0 填充）。"""
    frame_len = int(sample_rate * 0.03)
    hop = frame_len
    contour: List[float] = []
    min_lag = max(1, int(sample_rate / fmax))
    max_lag = int(sample_rate / fmin)
    for start in range(0, len(voiced) - frame_len + 1, hop):
        frame = voiced[start : start + frame_len].astype(np.float64)
        frame = frame - frame.mean()
        if np.abs(frame).max() < 1e-6:
            contour.append(0.0)
            continue
        corr = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        corr[:min_lag] = 0.0
        upper = min(max_lag, len(corr))
        if upper <= min_lag:
            contour.append(0.0)
            continue
        lag = int(np.argmax(corr[min_lag:upper])) + min_lag
        contour.append(float(sample_rate / lag))
    return contour


class AudioAnalyzer:
    """音訊特徵分析器（純 CPU）。"""

    def __init__(self, sample_rate: int = 16000) -> None:
        """初始化分析器。

        Args:
            sample_rate: 取樣率，預設 16000。

        Raises:
            ValueError: 取樣率非正數時。
        """
        if sample_rate <= 0:
            raise ValueError("sample_rate 必須為正整數")
        self.sample_rate = sample_rate

    def analyze(self, audio_array: np.ndarray) -> AudioFeatures:
        """分析音訊並回傳特徵。

        Args:
            audio_array: 單聲道波形（一維），多聲道會先取平均。

        Returns:
            :class:`AudioFeatures`。

        Raises:
            ValueError: 輸入為空或非數值時。
        """
        waveform = np.asarray(audio_array, dtype=np.float64)
        if waveform.size == 0:
            raise ValueError("音訊陣列不可為空")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        waveform = waveform.ravel()
        if not np.all(np.isfinite(waveform)):
            raise ValueError("音訊陣列含非有限值")
        sr = self.sample_rate
        duration = len(waveform) / sr
        energy = float(np.sqrt(np.mean(waveform**2)))

        frame_len = max(1, int(sr * 0.02))
        hop = max(1, frame_len // 2)
        frame_e = _frame_energy(waveform, frame_len, hop)
        median_e = float(np.median(frame_e)) if frame_e.size else 0.0
        threshold = max(median_e * 0.3, 1e-4)
        voiced_mask = frame_e > threshold

        pauses: List[Tuple[float, float]] = []
        in_pause = False
        pause_start = 0.0
        for i, voiced in enumerate(voiced_mask):
            t = i * hop / sr
            if not voiced and not in_pause:
                in_pause = True
                pause_start = t
            elif voiced and in_pause:
                in_pause = False
                if t - pause_start >= 0.15:
                    pauses.append((round(pause_start, 3), round(t, 3)))
        if in_pause and duration - pause_start >= 0.15:
            pauses.append((round(pause_start, 3), round(duration, 3)))

        voiced_ratio = float(voiced_mask.mean()) if voiced_mask.size else 0.0
        voiced_wave = waveform if voiced_ratio > 0.9 else waveform.copy()
        contour = _pitch_autocorr(voiced_wave, sr)
        voiced_pitch = [p for p in contour if p > 0]
        pitch_mean = float(np.mean(voiced_pitch)) if voiced_pitch else 0.0
        pitch_var = float(np.var(voiced_pitch)) if voiced_pitch else 0.0
        if len(voiced_pitch) >= 2:
            periods = [1.0 / p for p in voiced_pitch]
            jitter = float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-9))
        else:
            jitter = 0.0
        if frame_e.size >= 2 and float(np.mean(frame_e)) > 0:
            shimmer = float(np.mean(np.abs(np.diff(frame_e))) / (np.mean(frame_e) + 1e-9))
        else:
            shimmer = 0.0

        onsets = int(np.sum((frame_e[1:] > threshold) & (frame_e[:-1] <= threshold))) if frame_e.size >= 2 else 0
        speech_rate = round(onsets / duration, 2) if duration > 0 else 0.0

        spectrum = np.abs(np.fft.rfft(waveform * np.hanning(len(waveform))))
        freqs = np.fft.rfftfreq(len(waveform), d=1.0 / sr)
        total = float(spectrum.sum()) + 1e-12
        centroid = float((freqs * spectrum).sum() / total)
        bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * spectrum).sum() / total))

        return AudioFeatures(
            pitch_mean=round(pitch_mean, 2),
            pitch_variance=round(pitch_var, 2),
            pitch_contour=[round(p, 2) for p in contour],
            energy=round(energy, 4),
            speech_rate=speech_rate,
            pause_segments=pauses,
            jitter=round(jitter, 4),
            shimmer=round(shimmer, 4),
            spectral_features={
                "centroid": round(centroid, 2),
                "bandwidth": round(bandwidth, 2),
            },
            duration=round(duration, 3),
        )
