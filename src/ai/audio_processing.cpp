#include "../../include/ai/audio_processing.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <fstream>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace asekioml {
namespace ai {

// ===== Audio Loading and Saving =====

Tensor AudioProcessor::load_audio(const std::string& filepath, int sample_rate, int channels) {
    // Placeholder implementation - generates a test sine wave
    // TODO: Implement actual audio loading with third-party libraries (e.g., libsndfile, FMOD, OpenAL)
    
    // Default parameters
    int actual_sample_rate = (sample_rate == 0) ? 44100 : sample_rate;
    int actual_channels = (channels == 0) ? 1 : channels;
    double duration = 3.0; // 3 seconds
    
    size_t num_samples = static_cast<size_t>(actual_sample_rate * duration);
    
    std::vector<size_t> shape;
    if (actual_channels == 1) {
        shape = {num_samples};
    } else {
        shape = {num_samples, static_cast<size_t>(actual_channels)};
    }
    
    Tensor audio(shape);
    
    // Generate test sine wave at 440 Hz (A4 note)
    double frequency = 440.0;
    for (size_t i = 0; i < num_samples; ++i) {
        double t = static_cast<double>(i) / actual_sample_rate;
        double sample = 0.3 * std::sin(2.0 * M_PI * frequency * t);
        
        if (actual_channels == 1) {
            audio({i}) = sample;
        } else {
            for (int c = 0; c < actual_channels; ++c) {
                // Add slight phase offset for stereo effect
                double phase_offset = (c == 1) ? 0.1 : 0.0;
                double stereo_sample = 0.3 * std::sin(2.0 * M_PI * frequency * t + phase_offset);
                audio({i, static_cast<size_t>(c)}) = stereo_sample;
            }
        }
    }
    
    return audio;
}

void AudioProcessor::save_audio(const Tensor& audio, const std::string& filepath, 
                               int sample_rate, AudioFormat format) {
    // Placeholder implementation - logs the save operation
    // TODO: Implement actual audio saving with third-party libraries
    
    auto shape = audio.shape();
    size_t samples = shape[0];
    size_t channels = (shape.size() == 1) ? 1 : shape[1];
    
    // Calculate basic statistics
    double min_val = audio.data()[0];
    double max_val = audio.data()[0];
    double sum = 0.0;
    double rms = 0.0;
    
    for (size_t i = 0; i < audio.size(); ++i) {
        double val = audio.data()[i];
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        sum += val;
        rms += val * val;
    }
    double mean = sum / audio.size();
    rms = std::sqrt(rms / audio.size());
    
    // Duration calculation
    double duration = static_cast<double>(samples) / sample_rate;
    
    // Log the operation (in a real implementation, this would save the actual file)
    std::cout << "AudioProcessor::save_audio() - Placeholder Implementation\n";
    std::cout << "  File: " << filepath << "\n";
    std::cout << "  Duration: " << duration << " seconds\n";
    std::cout << "  Samples: " << samples << " (" << sample_rate << " Hz)\n";
    std::cout << "  Channels: " << channels << "\n";
    std::cout << "  Format: " << static_cast<int>(format) << "\n";
    std::cout << "  Amplitude range: [" << min_val << ", " << max_val << "] (RMS: " << rms << ")\n";
    std::cout << "  Note: Actual file saving not implemented yet\n";
}

// ===== Basic Audio Operations =====

Tensor AudioProcessor::to_mono(const Tensor& audio) {
    auto shape = audio.shape();
    if (shape.size() == 1) {
        return audio; // Already mono
    }
    
    if (shape.size() != 2) {
        throw std::invalid_argument("Audio tensor must be 1D or 2D [samples] or [samples, channels]");
    }
    
    size_t samples = shape[0];
    size_t channels = shape[1];
    
    Tensor mono({samples});
      for (size_t i = 0; i < samples; ++i) {
        double sum = 0.0;
        for (size_t c = 0; c < channels; ++c) {
            sum += audio({i, c});
        }
        mono({i}) = sum / channels;
    }
    
    return mono;
}

Tensor AudioProcessor::to_stereo(const Tensor& audio) {
    auto shape = audio.shape();
    if (shape.size() != 1) {
        throw std::invalid_argument("Input audio must be mono (1D tensor)");
    }
    
    size_t samples = shape[0];
    Tensor stereo({samples, 2});
      for (size_t i = 0; i < samples; ++i) {
        double value = audio({i});
        stereo({i, 0}) = value;
        stereo({i, 1}) = value;
    }
    
    return stereo;
}

Tensor AudioProcessor::resample(const Tensor& audio, int current_rate, int target_rate) {
    if (current_rate == target_rate) {
        return audio;
    }
    
    auto shape = audio.shape();
    if (shape.size() != 1) {
        throw std::invalid_argument("Currently only supports mono audio resampling");
    }
    
    size_t current_samples = shape[0];
    size_t target_samples = static_cast<size_t>((double)current_samples * target_rate / current_rate);
    
    Tensor resampled({target_samples});
    
    // Simple linear interpolation resampling
    for (size_t i = 0; i < target_samples; ++i) {
        double pos = (double)i * current_samples / target_samples;
        size_t idx = static_cast<size_t>(pos);
        double frac = pos - idx;
          if (idx >= current_samples - 1) {
            resampled({i}) = audio({current_samples - 1});
        } else {
            double val = audio({idx}) * (1.0 - frac) + audio({idx + 1}) * frac;
            resampled({i}) = val;
        }
    }
    
    return resampled;
}

Tensor AudioProcessor::normalize(const Tensor& audio, double target_peak) {
    auto shape = audio.shape();
    Tensor normalized = audio;
    
    // Find peak amplitude
    double max_val = 0.0;
    size_t total_elements = 1;
    for (size_t dim : shape) {
        total_elements *= dim;
    }
    
    for (size_t i = 0; i < total_elements; ++i) {
        std::vector<size_t> indices;
        size_t temp = i;
        for (int j = shape.size() - 1; j >= 0; --j) {
            indices.insert(indices.begin(), temp % shape[j]);
            temp /= shape[j];
        }
        max_val = std::max(max_val, std::abs(audio(indices)));
    }
    
    if (max_val > 0.0) {
        double scale = target_peak / max_val;
        normalized = normalized * scale;
    }
    
    return normalized;
}

// ===== FFT and Spectral Analysis =====

std::vector<double> AudioProcessor::generate_window(size_t size, WindowFunction func) {
    std::vector<double> window(size);
    
    switch (func) {
        case WindowFunction::NONE:
            std::fill(window.begin(), window.end(), 1.0);
            break;
            
        case WindowFunction::HANNING:
            for (size_t i = 0; i < size; ++i) {
                window[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (size - 1)));
            }
            break;
            
        case WindowFunction::HAMMING:
            for (size_t i = 0; i < size; ++i) {
                window[i] = 0.54 - 0.46 * std::cos(2.0 * M_PI * i / (size - 1));
            }
            break;
            
        case WindowFunction::BLACKMAN:
            for (size_t i = 0; i < size; ++i) {
                double n = i / (double)(size - 1);
                window[i] = 0.42 - 0.5 * std::cos(2.0 * M_PI * n) + 0.08 * std::cos(4.0 * M_PI * n);
            }
            break;
            
        case WindowFunction::GAUSSIAN:
            double sigma = 0.4;
            for (size_t i = 0; i < size; ++i) {
                double n = (i - (size - 1) / 2.0) / ((size - 1) / 2.0);
                window[i] = std::exp(-0.5 * (n / sigma) * (n / sigma));
            }
            break;
    }
    
    return window;
}

std::vector<double> AudioProcessor::apply_window(const std::vector<double>& frame, const std::vector<double>& window) {
    if (frame.size() != window.size()) {
        throw std::invalid_argument("Frame and window must have same size");
    }
    
    std::vector<double> windowed(frame.size());
    for (size_t i = 0; i < frame.size(); ++i) {
        windowed[i] = frame[i] * window[i];
    }
    
    return windowed;
}

// Simple FFT implementation (for educational purposes - in production, use FFTW or similar)
std::vector<std::complex<double>> AudioProcessor::fft(const std::vector<double>& signal) {
    size_t N = signal.size();
    
    // Ensure N is power of 2 (pad with zeros if necessary)
    size_t N_padded = 1;
    while (N_padded < N) N_padded *= 2;
    
    std::vector<std::complex<double>> x(N_padded);
    for (size_t i = 0; i < N; ++i) {
        x[i] = std::complex<double>(signal[i], 0.0);
    }
    for (size_t i = N; i < N_padded; ++i) {
        x[i] = std::complex<double>(0.0, 0.0);
    }
    
    // Cooley-Tukey FFT (simplified)
    std::vector<std::complex<double>> result(N_padded);
    for (size_t k = 0; k < N_padded; ++k) {
        std::complex<double> sum(0.0, 0.0);
        for (size_t n = 0; n < N_padded; ++n) {
            double angle = -2.0 * M_PI * k * n / N_padded;
            std::complex<double> twiddle(std::cos(angle), std::sin(angle));
            sum += x[n] * twiddle;
        }
        result[k] = sum;
    }
    
    return result;
}

std::vector<double> AudioProcessor::ifft(const std::vector<std::complex<double>>& spectrum) {
    size_t N = spectrum.size();
    
    // Conjugate the complex numbers
    std::vector<std::complex<double>> x(N);
    for (size_t i = 0; i < N; ++i) {
        x[i] = std::conj(spectrum[i]);
    }
    
    // Apply FFT
    std::vector<std::complex<double>> result(N);
    for (size_t k = 0; k < N; ++k) {
        std::complex<double> sum(0.0, 0.0);
        for (size_t n = 0; n < N; ++n) {
            double angle = -2.0 * M_PI * k * n / N;
            std::complex<double> twiddle(std::cos(angle), std::sin(angle));
            sum += x[n] * twiddle;
        }
        result[k] = sum;
    }
    
    // Conjugate again and scale
    std::vector<double> real_result(N);
    for (size_t i = 0; i < N; ++i) {
        real_result[i] = std::conj(result[i]).real() / N;
    }
    
    return real_result;
}

Tensor AudioProcessor::stft(const Tensor& audio, size_t window_size, size_t hop_size, WindowFunction window_func) {
    auto shape = audio.shape();
    if (shape.size() != 1) {
        throw std::invalid_argument("Audio must be 1D tensor for STFT");
    }
    
    size_t samples = shape[0];
    size_t num_frames = (samples - window_size) / hop_size + 1;
    size_t fft_size = window_size / 2 + 1; // Only positive frequencies
    
    auto window = generate_window(window_size, window_func);
    
    // For simplicity, we'll store magnitude in the tensor (not complex)
    Tensor stft_result({num_frames, fft_size});
    
    for (size_t frame = 0; frame < num_frames; ++frame) {
        size_t start = frame * hop_size;
          // Extract frame
        std::vector<double> frame_data(window_size);
        for (size_t i = 0; i < window_size; ++i) {
            if (start + i < samples) {
                frame_data[i] = audio({start + i});
            } else {
                frame_data[i] = 0.0;
            }
        }
        
        // Apply window
        auto windowed = apply_window(frame_data, window);
        
        // Compute FFT
        auto fft_result = fft(windowed);
        
        // Store magnitude spectrum (only positive frequencies)
        for (size_t bin = 0; bin < fft_size; ++bin) {
            double magnitude = std::abs(fft_result[bin]);
            stft_result({frame, bin}) = magnitude;
        }
    }
    
    return stft_result;
}

Tensor AudioProcessor::magnitude_spectrogram(const Tensor& stft_tensor) {
    // Since our STFT already returns magnitude, just return it
    return stft_tensor;
}

Tensor AudioProcessor::power_spectrogram(const Tensor& stft_tensor) {
    // Convert magnitude to power (square the values)
    auto shape = stft_tensor.shape();
    Tensor power_spec({shape[0], shape[1]});
      for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            double magnitude = stft_tensor({i, j});
            power_spec({i, j}) = magnitude * magnitude;
        }
    }
    
    return power_spec;
}

// ===== Mel Scale Conversion =====

double AudioProcessor::hz_to_mel(double freq) {
    return 2595.0 * std::log10(1.0 + freq / 700.0);
}

double AudioProcessor::mel_to_hz(double mel) {
    return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

Tensor AudioProcessor::create_mel_filter_bank(size_t n_filters, size_t fft_size, int sample_rate, double fmin, double fmax) {
    size_t n_bins = fft_size / 2 + 1;
    
    // Create mel filter bank
    Tensor filter_bank({n_filters, n_bins});
    
    // Convert frequencies to mel scale
    double mel_min = hz_to_mel(fmin);
    double mel_max = hz_to_mel(fmax);
    
    // Create mel points
    std::vector<double> mel_points(n_filters + 2);
    for (size_t i = 0; i < n_filters + 2; ++i) {
        mel_points[i] = mel_min + i * (mel_max - mel_min) / (n_filters + 1);
    }
    
    // Convert back to Hz
    std::vector<double> hz_points(n_filters + 2);
    for (size_t i = 0; i < n_filters + 2; ++i) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }
    
    // Convert to bin indices
    std::vector<size_t> bin_points(n_filters + 2);
    for (size_t i = 0; i < n_filters + 2; ++i) {
        bin_points[i] = static_cast<size_t>(hz_points[i] * fft_size / sample_rate);
    }
    
    // Create triangular filters
    for (size_t m = 0; m < n_filters; ++m) {
        size_t left = bin_points[m];
        size_t center = bin_points[m + 1];
        size_t right = bin_points[m + 2];
        
        for (size_t k = 0; k < n_bins; ++k) {
            double value = 0.0;
            
            if (k >= left && k <= center && center > left) {
                value = (double)(k - left) / (center - left);
            } else if (k >= center && k <= right && right > center) {
                value = (double)(right - k) / (right - center);
            }
            
            filter_bank({m, k}) = value;
        }
    }
    
    return filter_bank;
}

Tensor AudioProcessor::mel_spectrogram(const Tensor& power_spec, int sample_rate, size_t n_mels, double fmin, double fmax) {
    auto shape = power_spec.shape();
    size_t time_frames = shape[0];
    size_t freq_bins = shape[1];
    size_t fft_size = (freq_bins - 1) * 2; // Reconstruct FFT size
    
    // Create mel filter bank
    auto mel_filters = create_mel_filter_bank(n_mels, fft_size, sample_rate, fmin, fmax);
    
    // Apply mel filters
    Tensor mel_spec({time_frames, n_mels});
    
    for (size_t t = 0; t < time_frames; ++t) {        for (size_t m = 0; m < n_mels; ++m) {
            double sum = 0.0;
            for (size_t f = 0; f < freq_bins; ++f) {
                sum += power_spec({t, f}) * mel_filters({m, f});
            }
            mel_spec({t, m}) = sum;
        }
    }
    
    return mel_spec;
}

// ===== Audio Generation =====

Tensor AudioProcessor::generate_sine_wave(double frequency, double duration, int sample_rate, double amplitude) {
    size_t samples = static_cast<size_t>(duration * sample_rate);
    Tensor sine_wave({samples});
      for (size_t i = 0; i < samples; ++i) {
        double t = (double)i / sample_rate;
        double value = amplitude * std::sin(2.0 * M_PI * frequency * t);
        sine_wave({i}) = value;
    }
    
    return sine_wave;
}

Tensor AudioProcessor::generate_white_noise(double duration, int sample_rate, double amplitude) {
    size_t samples = static_cast<size_t>(duration * sample_rate);
    Tensor noise({samples});
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, amplitude);
      for (size_t i = 0; i < samples; ++i) {
        noise({i}) = dist(gen);
    }
    
    return noise;
}

Tensor AudioProcessor::generate_chirp(double start_freq, double end_freq, double duration, int sample_rate, double amplitude) {
    size_t samples = static_cast<size_t>(duration * sample_rate);
    Tensor chirp({samples});
    
    double freq_change = (end_freq - start_freq) / duration;
      for (size_t i = 0; i < samples; ++i) {
        double t = (double)i / sample_rate;
        double instantaneous_freq = start_freq + freq_change * t;
        double phase = 2.0 * M_PI * (start_freq * t + 0.5 * freq_change * t * t);
        double value = amplitude * std::sin(phase);
        chirp({i}) = value;
    }
    
    return chirp;
}

// ===== Batch Operations (Template implementation in header) =====

std::vector<Tensor> AudioProcessor::batch_mel_spectrogram(const std::vector<Tensor>& audio_batch,
                                                         int sample_rate,
                                                         size_t n_mels,
                                                         size_t window_size,
                                                         size_t hop_size) {
    std::vector<Tensor> mel_specs;
    mel_specs.reserve(audio_batch.size());
    
    for (const auto& audio : audio_batch) {
        // STFT -> Power Spectrogram -> Mel Spectrogram
        auto stft_result = stft(audio, window_size, hop_size);
        auto power_spec = power_spectrogram(stft_result);
        auto mel_spec = mel_spectrogram(power_spec, sample_rate, n_mels);
        mel_specs.push_back(mel_spec);
    }
    
    return mel_specs;
}

// ===== Audio Filters (Basic implementations) =====

Tensor AudioProcessor::low_pass_filter(const Tensor& audio, double cutoff_freq, int sample_rate) {
    // Simple IIR low-pass filter implementation
    // In practice, you'd want a more sophisticated filter design
    
    auto shape = audio.shape();
    if (shape.size() != 1) {
        throw std::invalid_argument("Currently only supports mono audio filtering");
    }
    
    Tensor filtered = audio;
    double RC = 1.0 / (2.0 * M_PI * cutoff_freq);
    double dt = 1.0 / sample_rate;
    double alpha = dt / (RC + dt);
      // Apply filter: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
    for (size_t i = 1; i < shape[0]; ++i) {
        double current = audio({i});
        double previous = filtered({i-1});
        filtered({i}) = alpha * current + (1.0 - alpha) * previous;
    }
    
    return filtered;
}

Tensor AudioProcessor::high_pass_filter(const Tensor& audio, double cutoff_freq, int sample_rate) {
    // High-pass = Original - Low-pass
    auto low_passed = low_pass_filter(audio, cutoff_freq, sample_rate);
    return audio - low_passed;
}

Tensor AudioProcessor::band_pass_filter(const Tensor& audio, double low_freq, double high_freq, int sample_rate) {
    // Band-pass = High-pass(low_freq) - High-pass(high_freq)
    auto high_passed = high_pass_filter(audio, low_freq, sample_rate);
    auto low_passed = low_pass_filter(high_passed, high_freq, sample_rate);
    return low_passed;
}

Tensor AudioProcessor::add_reverb(const Tensor& audio, double room_size, double damping, double wet_level) {
    // Simple reverb implementation using delay lines
    auto shape = audio.shape();
    if (shape.size() != 1) {
        throw std::invalid_argument("Currently only supports mono audio reverb");
    }
    
    size_t samples = shape[0];
    Tensor reverb_audio = audio;
    
    // Simple delay-based reverb with multiple delay lines
    std::vector<size_t> delays = {
        static_cast<size_t>(0.03 * 44100),  // ~30ms
        static_cast<size_t>(0.05 * 44100),  // ~50ms
        static_cast<size_t>(0.07 * 44100),  // ~70ms
        static_cast<size_t>(0.11 * 44100)   // ~110ms
    };
      for (auto delay : delays) {
        for (size_t i = delay; i < samples; ++i) {
            double delayed_sample = audio({i - delay}) * room_size * (1.0 - damping);
            double current = reverb_audio({i});
            reverb_audio({i}) = current + delayed_sample * wet_level;
        }
    }
    
    return reverb_audio;
}

} // namespace ai
} // namespace asekioml
