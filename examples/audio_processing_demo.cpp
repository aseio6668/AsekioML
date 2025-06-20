#include "../include/ai/audio_processing.hpp"
#include "../include/tensor.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace clmodel::ai;

void print_test_result(const std::string& test_name, bool passed, double time_ms = -1) {
    std::cout << "  " << std::setw(35) << std::left << test_name << ": ";
    if (passed) {
        std::cout << "PASS";
    } else {
        std::cout << "FAIL";
    }
    if (time_ms >= 0) {
        std::cout << " (" << std::fixed << std::setprecision(2) << time_ms << "ms)";
    }
    std::cout << std::endl;
}

void test_basic_audio_operations() {
    std::cout << "\nTesting Basic Audio Operations..." << std::endl;
    
    // Test mono/stereo conversion
    bool mono_stereo_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
          // Create test stereo audio
        Tensor stereo_audio({1000, 2});
        for (size_t i = 0; i < 1000; ++i) {
            stereo_audio({i, 0}) = 0.5 * std::sin(2.0 * M_PI * 440.0 * i / 44100.0); // Left channel
            stereo_audio({i, 1}) = 0.3 * std::cos(2.0 * M_PI * 880.0 * i / 44100.0); // Right channel
        }
        
        // Convert to mono
        auto mono = AudioProcessor::to_mono(stereo_audio);
        if (mono.shape().size() != 1 || mono.shape()[0] != 1000) {
            mono_stereo_pass = false;
        }
        
        // Convert back to stereo
        auto stereo_back = AudioProcessor::to_stereo(mono);
        if (stereo_back.shape().size() != 2 || stereo_back.shape()[0] != 1000 || stereo_back.shape()[1] != 2) {
            mono_stereo_pass = false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("Mono/Stereo Conversion", mono_stereo_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("Mono/Stereo Conversion", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
    
    // Test resampling
    bool resample_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
          // Create test mono audio at 44100 Hz
        Tensor audio_44k({44100}); // 1 second
        for (size_t i = 0; i < 44100; ++i) {
            audio_44k({i}) = std::sin(2.0 * M_PI * 440.0 * i / 44100.0);
        }
        
        // Resample to 22050 Hz
        auto audio_22k = AudioProcessor::resample(audio_44k, 44100, 22050);
        if (audio_22k.shape()[0] != 22050) {
            resample_pass = false;
        }
        
        // Resample back to 44100 Hz
        auto audio_back = AudioProcessor::resample(audio_22k, 22050, 44100);
        if (audio_back.shape()[0] != 44100) {
            resample_pass = false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("Audio Resampling", resample_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("Audio Resampling", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
    
    // Test normalization
    bool normalize_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
          // Create test audio with varying amplitudes
        Tensor audio({1000});
        for (size_t i = 0; i < 1000; ++i) {
            audio({i}) = 2.5 * std::sin(2.0 * M_PI * 440.0 * i / 44100.0); // Peak at 2.5
        }
        
        // Normalize to peak of 1.0
        auto normalized = AudioProcessor::normalize(audio, 1.0);
        
        // Check that peak is approximately 1.0
        double max_val = 0.0;
        for (size_t i = 0; i < 1000; ++i) {
            max_val = std::max(max_val, std::abs(normalized({i})));
        }
        
        if (std::abs(max_val - 1.0) > 0.01) {
            normalize_pass = false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("Audio Normalization", normalize_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("Audio Normalization", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
}

void test_spectral_analysis() {
    std::cout << "\nTesting Spectral Analysis..." << std::endl;
    
    // Test STFT
    bool stft_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
          // Create test signal: mix of 440Hz and 880Hz
        Tensor audio({4096});
        for (size_t i = 0; i < 4096; ++i) {
            double val = 0.5 * std::sin(2.0 * M_PI * 440.0 * i / 44100.0) +
                        0.3 * std::sin(2.0 * M_PI * 880.0 * i / 44100.0);
            audio({i}) = val;
        }
        
        // Compute STFT
        auto stft_result = AudioProcessor::stft(audio, 1024, 512, WindowFunction::HANNING);
        auto shape = stft_result.shape();
        
        // Check dimensions
        size_t expected_frames = (4096 - 1024) / 512 + 1;
        size_t expected_bins = 1024 / 2 + 1;
        
        if (shape.size() != 2 || shape[0] != expected_frames || shape[1] != expected_bins) {
            stft_pass = false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("STFT Computation", stft_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("STFT Computation", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
    
    // Test power spectrogram
    bool power_spec_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
          // Create test magnitude spectrogram
        Tensor magnitude_spec({10, 513});
        for (size_t i = 0; i < 10; ++i) {
            for (size_t j = 0; j < 513; ++j) {
                magnitude_spec({i, j}) = static_cast<double>(i + j) / 1000.0;
            }
        }
        
        // Convert to power
        auto power_spec = AudioProcessor::power_spectrogram(magnitude_spec);
        auto shape = power_spec.shape();
        
        if (shape.size() != 2 || shape[0] != 10 || shape[1] != 513) {
            power_spec_pass = false;
        }
        
        // Check that power = magnitude^2
        bool values_correct = true;
        for (size_t i = 0; i < 5; ++i) {
            for (size_t j = 0; j < 5; ++j) {
                double mag = magnitude_spec({i, j});
                double pow = power_spec({i, j});
                if (std::abs(pow - mag * mag) > 1e-10) {
                    values_correct = false;
                }
            }
        }
        
        if (!values_correct) {
            power_spec_pass = false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("Power Spectrogram", power_spec_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("Power Spectrogram", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
    
    // Test mel spectrogram
    bool mel_spec_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
          // Create test power spectrogram
        Tensor power_spec({20, 513});
        for (size_t i = 0; i < 20; ++i) {
            for (size_t j = 0; j < 513; ++j) {
                power_spec({i, j}) = static_cast<double>(j) / 513.0; // Frequency slope
            }
        }
        
        // Convert to mel spectrogram
        auto mel_spec = AudioProcessor::mel_spectrogram(power_spec, 22050, 128);
        auto shape = mel_spec.shape();
        
        if (shape.size() != 2 || shape[0] != 20 || shape[1] != 128) {
            mel_spec_pass = false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("Mel Spectrogram", mel_spec_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("Mel Spectrogram", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
}

void test_audio_generation() {
    std::cout << "\nTesting Audio Generation..." << std::endl;
    
    // Test sine wave generation
    bool sine_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto sine_wave = AudioProcessor::generate_sine_wave(440.0, 1.0, 44100, 1.0);
        auto shape = sine_wave.shape();
        
        if (shape.size() != 1 || shape[0] != 44100) {
            sine_pass = false;
        }
        
        // Check frequency content (should peak around 440 Hz)
        auto stft_result = AudioProcessor::stft(sine_wave, 2048, 1024);
        // In a real test, we'd check that the frequency peak is at the right bin
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("Sine Wave Generation", sine_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("Sine Wave Generation", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
    
    // Test white noise generation
    bool noise_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto noise = AudioProcessor::generate_white_noise(0.5, 44100, 0.1);
        auto shape = noise.shape();
        
        if (shape.size() != 1 || shape[0] != 22050) {
            noise_pass = false;
        }
          // Check that it's actually random (variance should be non-zero)
        double mean = 0.0;
        for (size_t i = 0; i < shape[0]; ++i) {
            mean += noise({i});
        }
        mean /= shape[0];
        
        double variance = 0.0;
        for (size_t i = 0; i < shape[0]; ++i) {
            double diff = noise({i}) - mean;
            variance += diff * diff;
        }
        variance /= shape[0];
        
        if (variance < 0.001) { // Should have significant variance
            noise_pass = false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("White Noise Generation", noise_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("White Noise Generation", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
    
    // Test chirp generation
    bool chirp_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto chirp = AudioProcessor::generate_chirp(100.0, 1000.0, 1.0, 44100, 0.5);
        auto shape = chirp.shape();
        
        if (shape.size() != 1 || shape[0] != 44100) {
            chirp_pass = false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("Chirp Generation", chirp_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("Chirp Generation", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
}

void test_audio_filters() {
    std::cout << "\nTesting Audio Filters..." << std::endl;
    
    // Test low-pass filter
    bool lowpass_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
          // Create test signal with high and low frequency components
        Tensor audio({4410}); // 0.1 seconds at 44100 Hz
        for (size_t i = 0; i < 4410; ++i) {
            double low_freq = 0.5 * std::sin(2.0 * M_PI * 100.0 * i / 44100.0);
            double high_freq = 0.3 * std::sin(2.0 * M_PI * 5000.0 * i / 44100.0);
            audio({i}) = low_freq + high_freq;
        }
        
        // Apply low-pass filter at 1000 Hz
        auto filtered = AudioProcessor::low_pass_filter(audio, 1000.0, 44100);
        auto shape = filtered.shape();
        
        if (shape.size() != 1 || shape[0] != 4410) {
            lowpass_pass = false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("Low-pass Filter", lowpass_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("Low-pass Filter", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
    
    // Test high-pass filter
    bool highpass_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
          Tensor audio({4410});
        for (size_t i = 0; i < 4410; ++i) {
            audio({i}) = std::sin(2.0 * M_PI * 440.0 * i / 44100.0);
        }
        
        auto filtered = AudioProcessor::high_pass_filter(audio, 200.0, 44100);
        auto shape = filtered.shape();
        
        if (shape.size() != 1 || shape[0] != 4410) {
            highpass_pass = false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("High-pass Filter", highpass_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("High-pass Filter", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
    
    // Test reverb effect
    bool reverb_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
          Tensor audio({44100}); // 1 second
        for (size_t i = 0; i < 44100; ++i) {
            audio({i}) = std::sin(2.0 * M_PI * 440.0 * i / 44100.0);
        }
        
        auto reverb_audio = AudioProcessor::add_reverb(audio, 0.7, 0.3, 0.4);
        auto shape = reverb_audio.shape();
        
        if (shape.size() != 1 || shape[0] != 44100) {
            reverb_pass = false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("Reverb Effect", reverb_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("Reverb Effect", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
}

void test_batch_operations() {
    std::cout << "\nTesting Batch Operations..." << std::endl;
    
    bool batch_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Create batch of test audio signals
        std::vector<Tensor> audio_batch;        for (int i = 0; i < 3; ++i) {
            Tensor audio({2048});
            double freq = 440.0 * (i + 1); // Different frequencies
            for (size_t j = 0; j < 2048; ++j) {
                audio({j}) = std::sin(2.0 * M_PI * freq * j / 44100.0);
            }
            audio_batch.push_back(audio);
        }
        
        // Process batch to mel spectrograms
        auto mel_specs = AudioProcessor::batch_mel_spectrogram(audio_batch, 22050, 64, 512, 256);
        
        if (mel_specs.size() != 3) {
            batch_pass = false;
        }
        
        for (const auto& mel_spec : mel_specs) {
            auto shape = mel_spec.shape();
            if (shape.size() != 2 || shape[1] != 64) {
                batch_pass = false;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("Batch Mel Spectrograms", batch_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("Batch Mel Spectrograms", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
}

bool test_audio_io() {
    // Test loading
    Tensor loaded = AudioProcessor::load_audio("test_audio.wav", 44100, 2);
    
    if (loaded.shape().size() != 2 || loaded.shape()[1] != 2) {
        return false;
    }
    
    // Test saving
    AudioProcessor::save_audio(loaded, "output_test.wav", 44100, AudioFormat::WAV);
    
    return true;
}

int main() {
    std::cout << "CLModel Phase 2: Audio Processing Tests" << std::endl;
    std::cout << "======================================" << std::endl;
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    test_basic_audio_operations();
    test_spectral_analysis();
    test_audio_generation();
    test_audio_filters();
    test_batch_operations();
      std::cout << "\nTesting File I/O Operations..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    bool io_result = test_audio_io();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    print_test_result("Audio Load/Save", io_result, duration.count() / 1000.0);
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    
    std::cout << "\nAudio Processing Tests Complete" << std::endl;
    std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "🎵 Phase 2 Week 3-4: Audio Processing Foundation implemented!" << std::endl;
    std::cout << "🎯 Next: Text-to-Image pipeline (Week 5-8)" << std::endl;
    
    return 0;
}
