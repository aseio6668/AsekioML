#include "../include/ai/text_to_speech.hpp"
#include "../include/ai/audio_processing.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

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

bool test_text_preprocessing() {
    try {
        std::string text = "Hello, World! How are you today?";
        std::string processed = TextToPhonemeConverter::preprocess_text(text);
        
        // Should be lowercase and cleaned
        if (processed.find("hello") == std::string::npos || processed.find("world") == std::string::npos) {
            return false;
        }
        
        // Should not contain punctuation
        if (processed.find(",") != std::string::npos || processed.find("!") != std::string::npos) {
            return false;
        }
        
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool test_phoneme_conversion() {
    try {
        std::string text = "hello world";
        auto phonemes = TextToPhonemeConverter::text_to_phonemes(text);
        
        // Should have multiple phonemes
        if (phonemes.size() < 5) {
            return false;
        }
        
        // Check that phonemes have valid properties
        for (const auto& phoneme : phonemes) {
            if (phoneme.duration <= 0.0 || phoneme.energy < 0.0) {
                return false;
            }
        }
        
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool test_vocoder_synthesis() {
    try {
        VoiceConfig config;
        config.sample_rate = 16000; // Lower sample rate for faster testing
        config.base_pitch = 150.0;
        
        NeuralVocoder vocoder(config);
        
        // Create simple phoneme sequence
        std::vector<Phoneme> phonemes;
        
        Phoneme p1;
        p1.symbol = "h";
        p1.duration = 0.1;
        p1.pitch = 150.0;
        p1.energy = 0.7;
        phonemes.push_back(p1);
        
        Phoneme p2;
        p2.symbol = "eh";
        p2.duration = 0.15;
        p2.pitch = 160.0;
        p2.energy = 0.8;
        phonemes.push_back(p2);
        
        Phoneme p3;
        p3.symbol = "l";
        p3.duration = 0.08;
        p3.pitch = 140.0;
        p3.energy = 0.6;
        phonemes.push_back(p3);
        
        auto audio = vocoder.synthesize_from_phonemes(phonemes);
        
        if (audio.shape().size() != 1 || audio.size() == 0) {
            return false;
        }
        
        // Check audio is not all zeros
        bool has_signal = false;
        for (size_t i = 0; i < audio.size(); ++i) {
            if (std::abs(audio.data()[i]) > 0.001) {
                has_signal = true;
                break;
            }
        }
        
        return has_signal;
    } catch (const std::exception&) {
        return false;
    }
}

bool test_tts_pipeline() {
    try {
        VoiceConfig config;
        config.sample_rate = 16000; // Lower sample rate for faster testing
        
        TextToSpeechPipeline pipeline(config);
        
        auto audio = pipeline.synthesize("hello");
        
        if (audio.shape().size() != 1 || audio.size() == 0) {
            return false;
        }
        
        // Check for reasonable duration (should be at least 0.2 seconds)
        double duration = static_cast<double>(audio.size()) / config.sample_rate;
        if (duration < 0.1 || duration > 2.0) {
            return false;
        }
        
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool test_batch_synthesis() {
    try {
        VoiceConfig config;
        config.sample_rate = 16000;
        
        TextToSpeechPipeline pipeline(config);
        
        std::vector<std::string> texts = {"hello", "world", "test"};
        auto audio_batch = pipeline.batch_synthesize(texts);
        
        if (audio_batch.size() != texts.size()) {
            return false;
        }
        
        for (const auto& audio : audio_batch) {
            if (audio.size() == 0) {
                return false;
            }
        }
        
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool test_voice_configuration() {
    try {
        VoiceConfig config1;
        config1.base_pitch = 100.0;
        config1.speaking_rate = 0.8;
        
        VoiceConfig config2;
        config2.base_pitch = 200.0;
        config2.speaking_rate = 1.2;
        
        TextToSpeechPipeline pipeline(config1);
        
        // Test initial config
        if (pipeline.get_voice_config().base_pitch != config1.base_pitch) {
            return false;
        }
        
        // Test config change
        pipeline.set_voice_config(config2);
        if (pipeline.get_voice_config().base_pitch != config2.base_pitch) {
            return false;
        }
        
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void run_text_to_speech_tests() {
    std::cout << "\nCLModel Phase 2: Text-to-Speech Pipeline Tests\n";
    std::cout << "===============================================\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "\nTesting Text Processing..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    bool preprocessing_result = test_text_preprocessing();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    print_test_result("Text Preprocessing", preprocessing_result, duration.count() / 1000.0);
    
    start = std::chrono::high_resolution_clock::now();
    bool phoneme_result = test_phoneme_conversion();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    print_test_result("Phoneme Conversion", phoneme_result, duration.count() / 1000.0);
    
    std::cout << "\nTesting Audio Synthesis..." << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    bool vocoder_result = test_vocoder_synthesis();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    print_test_result("Vocoder Synthesis", vocoder_result, duration.count() / 1000.0);
    
    std::cout << "\nTesting TTS Pipeline..." << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    bool pipeline_result = test_tts_pipeline();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    print_test_result("Pipeline Synthesis", pipeline_result, duration.count() / 1000.0);
    
    start = std::chrono::high_resolution_clock::now();
    bool batch_result = test_batch_synthesis();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    print_test_result("Batch Synthesis", batch_result, duration.count() / 1000.0);
    
    start = std::chrono::high_resolution_clock::now();
    bool config_result = test_voice_configuration();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    print_test_result("Voice Configuration", config_result, duration.count() / 1000.0);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\nText-to-Speech Pipeline Tests Complete" << std::endl;
    std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "🎙️ Phase 2 Week 9-12: Text-to-Speech Pipeline foundation implemented!" << std::endl;
    std::cout << "📝 Note: This is a simplified educational implementation" << std::endl;
    std::cout << "🎯 Next: Production polish and optimization" << std::endl;
}

void demonstrate_text_to_speech() {
    std::cout << "\n=== Text-to-Speech Synthesis Demo ===" << std::endl;
    std::cout << "Synthesizing speech for demonstration prompts..." << std::endl;
    
    VoiceConfig config;
    config.sample_rate = 22050;
    config.base_pitch = 150.0;
    config.speaking_rate = 1.0;
    
    TextToSpeechPipeline pipeline(config);
    
    std::vector<std::string> demo_texts = {
        "hello world",
        "artificial intelligence",
        "neural network synthesis",
        "good morning"
    };
    
    for (const auto& text : demo_texts) {
        std::cout << "\nText: \"" << text << "\"" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        auto audio = pipeline.synthesize(text);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Calculate audio statistics
        double min_val = audio.data()[0];
        double max_val = audio.data()[0];
        double rms = 0.0;
        
        for (size_t i = 0; i < audio.size(); ++i) {
            double val = audio.data()[i];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            rms += val * val;
        }
        rms = std::sqrt(rms / audio.size());
        
        double audio_duration = static_cast<double>(audio.size()) / config.sample_rate;
        
        std::cout << "Generated audio duration: " << std::fixed << std::setprecision(2) << audio_duration << " seconds" << std::endl;
        std::cout << "Synthesis time: " << duration.count() << " ms" << std::endl;
        std::cout << "Audio statistics - Range: [" << std::setprecision(4) << min_val 
                  << ", " << max_val << "], RMS: " << rms << std::endl;
        
        // Demonstrate saving (placeholder)
        AudioProcessor::save_audio(audio, "output_" + text + ".wav", config.sample_rate, AudioFormat::WAV);
    }
}

int main() {
    run_text_to_speech_tests();
    demonstrate_text_to_speech();
    return 0;
}
