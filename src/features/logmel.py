import torch
import torch.nn as nn
import torchaudio

class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80, f_min=0, f_max=8000):
        super(LogMelSpectrogram, self).__init__()
        
        # 1. Khởi tạo bộ chuyển đổi Mel Spectrogram
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0 # Power phổ (bình phương biên độ)
        )
        
        # 2. Khởi tạo bộ chuyển đổi biên độ sang Decibel (Log Scale)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        
    def forward(self, waveform):
        """
        Input: waveform - Tensor có shape (Batch, 1, Samples)
        Output: log_mel - Tensor có shape (Batch, 1, n_mels, TimeFrames)
        """
        # Bước 1: Trích xuất Mel Spectrogram
        mel = self.mel_spectrogram(waveform)
        
        # Bước 2: Chuyển sang thang đo Log (Decibel) để giống với cách tai người nghe
        log_mel = self.amplitude_to_db(mel)
        
        return log_mel

# Test thử xem chiều dữ liệu (Shape) có đúng chuẩn để nhét vào CNN không
if __name__ == "__main__":
    # Giả lập 1 batch gồm 2 file audio dài 3 giây (3s * 16000hz = 48000 samples)
    dummy_waveform = torch.randn(2, 1, 48000) 
    
    extractor = LogMelSpectrogram()
    log_mel = extractor(dummy_waveform)
    
    print(f"Input waveform shape: {dummy_waveform.shape} -> (Batch, Channel, Samples)")
    print(f"Output Log-Mel shape: {log_mel.shape} -> (Batch, Channel, Height, Width)")
    print(f"Height = {log_mel.shape[2]} (Số dải n_mels)")
    print(f"Width = {log_mel.shape[3]} (Số khung thời gian TimeFrames)")