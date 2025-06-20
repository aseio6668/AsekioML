#pragma once

#include "../tensor.hpp"
#include <string>
#include <vector>
#include <memory>
#include <complex>

namespace clmodel {
namespace ai {

/**
 * @brief Audio format definitions
 */
enum class AudioFormat {
    WAV,
    FLAC,
    MP3,
    OGG
};

/**
 * @brief Window functions for FFT and signal processing
 */
enum class WindowFunction {
    NONE,
    HANNING,
    HAMMING,
    BLACKMAN,
    GAUSSIAN
};

/**
 * @brief Advanced audio processing operations built on the Tensor class
 * 
 * Provides high-performance audio manipulation capabilities for AI pipelines,
 * including preprocessing for speech models and audio generation.
 */
class AudioProcessor {
public:
    // ===== Audio Loading and Saving =====
    
    /**
     * @brief Load audio from file to tensor
     * @param filepath Path to audio file (WAV, FLAC, etc.)
     * @param sample_rate Target sample rate for loading (0 = keep original)
     * @param channels Target channels (0 = keep original, 1 = mono, 2 = stereo)
     * @return Tensor with shape [samples] for mono or [samples, channels] for multi-channel
     */
    static Tensor load_audio(const std::string& filepath, 
                           int sample_rate = 0,
                           int channels = 0);
    
    /**
     * @brief Save tensor as audio file
     * @param audio Audio tensor [samples] or [samples, channels]
     * @param filepath Output file path
     * @param sample_rate Sample rate for output
     * @param format Audio format for output
     */
    static void save_audio(const Tensor& audio, 
                         const std::string& filepath,
                         int sample_rate,
                         AudioFormat format = AudioFormat::WAV);
    
    // ===== Basic Audio Operations =====
    
    /**
     * @brief Convert audio tensor to mono by averaging channels
     * @param audio Input audio tensor [samples, channels]
     * @return Mono audio tensor [samples]
     */
    static Tensor to_mono(const Tensor& audio);
    
    /**
     * @brief Convert mono audio to stereo by duplicating channels
     * @param audio Input mono audio tensor [samples]
     * @return Stereo audio tensor [samples, 2]
     */
    static Tensor to_stereo(const Tensor& audio);
    
    /**
     * @brief Resample audio to different sample rate
     * @param audio Input audio tensor
     * @param current_rate Current sample rate
     * @param target_rate Target sample rate
     * @return Resampled audio tensor
     */
    static Tensor resample(const Tensor& audio, int current_rate, int target_rate);
    
    /**
     * @brief Normalize audio amplitude
     * @param audio Input audio tensor
     * @param target_peak Target peak amplitude (default: 1.0)
     * @return Normalized audio tensor
     */
    static Tensor normalize(const Tensor& audio, double target_peak = 1.0);
    
    // ===== FFT and Spectral Analysis =====
    
    /**
     * @brief Compute Short-Time Fourier Transform (STFT)
     * @param audio Input audio tensor [samples]
     * @param window_size Size of FFT window
     * @param hop_size Hop size between windows
     * @param window_func Window function to apply
     * @return Complex STFT tensor [time_frames, frequency_bins]
     */
    static Tensor stft(const Tensor& audio,
                      size_t window_size = 2048,
                      size_t hop_size = 512,
                      WindowFunction window_func = WindowFunction::HANNING);
    
    /**
     * @brief Compute inverse Short-Time Fourier Transform
     * @param stft_tensor Complex STFT tensor [time_frames, frequency_bins]
     * @param window_size Size of FFT window used in forward transform
     * @param hop_size Hop size used in forward transform
     * @param window_func Window function used in forward transform
     * @return Reconstructed audio tensor [samples]
     */
    static Tensor istft(const Tensor& stft_tensor,
                       size_t window_size = 2048,
                       size_t hop_size = 512,
                       WindowFunction window_func = WindowFunction::HANNING);
    
    /**
     * @brief Convert STFT to magnitude spectrogram
     * @param stft_tensor Complex STFT tensor
     * @return Magnitude spectrogram [time_frames, frequency_bins]
     */
    static Tensor magnitude_spectrogram(const Tensor& stft_tensor);
    
    /**
     * @brief Convert STFT to power spectrogram
     * @param stft_tensor Complex STFT tensor
     * @return Power spectrogram [time_frames, frequency_bins]
     */
    static Tensor power_spectrogram(const Tensor& stft_tensor);
    
    /**
     * @brief Convert power spectrogram to mel spectrogram
     * @param power_spec Power spectrogram tensor
     * @param sample_rate Sample rate of original audio
     * @param n_mels Number of mel frequency bins
     * @param fmin Minimum frequency
     * @param fmax Maximum frequency
     * @return Mel spectrogram [time_frames, n_mels]
     */
    static Tensor mel_spectrogram(const Tensor& power_spec,
                                 int sample_rate = 22050,
                                 size_t n_mels = 128,
                                 double fmin = 0.0,
                                 double fmax = 11025.0);
    
    // ===== Audio Filters and Effects =====
    
    /**
     * @brief Apply low-pass filter
     * @param audio Input audio tensor
     * @param cutoff_freq Cutoff frequency
     * @param sample_rate Sample rate
     * @return Filtered audio tensor
     */
    static Tensor low_pass_filter(const Tensor& audio, double cutoff_freq, int sample_rate);
    
    /**
     * @brief Apply high-pass filter
     * @param audio Input audio tensor
     * @param cutoff_freq Cutoff frequency
     * @param sample_rate Sample rate
     * @return Filtered audio tensor
     */
    static Tensor high_pass_filter(const Tensor& audio, double cutoff_freq, int sample_rate);
    
    /**
     * @brief Apply band-pass filter
     * @param audio Input audio tensor
     * @param low_freq Low cutoff frequency
     * @param high_freq High cutoff frequency
     * @param sample_rate Sample rate
     * @return Filtered audio tensor
     */
    static Tensor band_pass_filter(const Tensor& audio, double low_freq, double high_freq, int sample_rate);
    
    /**
     * @brief Add reverb effect
     * @param audio Input audio tensor
     * @param room_size Room size parameter (0.0 to 1.0)
     * @param damping Damping parameter (0.0 to 1.0)
     * @param wet_level Wet signal level (0.0 to 1.0)
     * @return Audio with reverb effect
     */
    static Tensor add_reverb(const Tensor& audio, double room_size = 0.5, double damping = 0.5, double wet_level = 0.3);
    
    // ===== Batch Operations =====
    
    /**
     * @brief Process multiple audio files in batch
     * @param audio_batch Batch of audio tensors [batch_size, samples] or [batch_size, samples, channels]
     * @param operation Lambda function to apply to each audio tensor
     * @return Batch of processed audio tensors
     */
    template<typename Func>
    static std::vector<Tensor> batch_process(const std::vector<Tensor>& audio_batch, Func operation);
    
    /**
     * @brief Create mel spectrograms for a batch of audio
     * @param audio_batch Batch of audio tensors
     * @param sample_rate Sample rate
     * @param n_mels Number of mel bins
     * @param window_size STFT window size
     * @param hop_size STFT hop size
     * @return Batch of mel spectrograms [batch_size, time_frames, n_mels]
     */
    static std::vector<Tensor> batch_mel_spectrogram(const std::vector<Tensor>& audio_batch,
                                                    int sample_rate = 22050,
                                                    size_t n_mels = 128,
                                                    size_t window_size = 2048,
                                                    size_t hop_size = 512);
    
    // ===== Audio Generation Utilities =====
    
    /**
     * @brief Generate sine wave
     * @param frequency Frequency in Hz
     * @param duration Duration in seconds
     * @param sample_rate Sample rate
     * @param amplitude Amplitude (0.0 to 1.0)
     * @return Generated sine wave tensor
     */
    static Tensor generate_sine_wave(double frequency, double duration, int sample_rate = 44100, double amplitude = 1.0);
    
    /**
     * @brief Generate white noise
     * @param duration Duration in seconds
     * @param sample_rate Sample rate
     * @param amplitude Amplitude (0.0 to 1.0)
     * @return Generated white noise tensor
     */
    static Tensor generate_white_noise(double duration, int sample_rate = 44100, double amplitude = 1.0);
    
    /**
     * @brief Generate chirp signal (frequency sweep)
     * @param start_freq Starting frequency
     * @param end_freq Ending frequency
     * @param duration Duration in seconds
     * @param sample_rate Sample rate
     * @param amplitude Amplitude (0.0 to 1.0)
     * @return Generated chirp signal tensor
     */
    static Tensor generate_chirp(double start_freq, double end_freq, double duration, int sample_rate = 44100, double amplitude = 1.0);

private:
    // ===== Helper Functions =====
    
    /**
     * @brief Generate window function
     * @param size Window size
     * @param func Window function type
     * @return Window coefficients
     */
    static std::vector<double> generate_window(size_t size, WindowFunction func);
    
    /**
     * @brief Apply window function to audio frame
     * @param frame Audio frame
     * @param window Window coefficients
     * @return Windowed frame
     */
    static std::vector<double> apply_window(const std::vector<double>& frame, const std::vector<double>& window);
    
    /**
     * @brief Compute FFT of real signal
     * @param signal Input real signal
     * @return Complex FFT result
     */
    static std::vector<std::complex<double>> fft(const std::vector<double>& signal);
    
    /**
     * @brief Compute IFFT of complex signal
     * @param spectrum Input complex spectrum
     * @return Real IFFT result
     */
    static std::vector<double> ifft(const std::vector<std::complex<double>>& spectrum);
    
    /**
     * @brief Convert frequency to mel scale
     * @param freq Frequency in Hz
     * @return Mel scale value
     */
    static double hz_to_mel(double freq);
    
    /**
     * @brief Convert mel scale to frequency
     * @param mel Mel scale value
     * @return Frequency in Hz
     */
    static double mel_to_hz(double mel);
    
    /**
     * @brief Create mel filter bank
     * @param n_filters Number of mel filters
     * @param fft_size Size of FFT
     * @param sample_rate Sample rate
     * @param fmin Minimum frequency
     * @param fmax Maximum frequency
     * @return Mel filter bank matrix [n_filters, fft_size/2 + 1]
     */
    static Tensor create_mel_filter_bank(size_t n_filters, size_t fft_size, int sample_rate, double fmin, double fmax);
};

} // namespace ai
} // namespace clmodel
