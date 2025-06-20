#include "../../include/ai/text_to_speech.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <regex>
#include <sstream>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace asekioml {
namespace ai {

// ===== TextToPhonemeConverter Implementation =====

std::unordered_map<std::string, std::vector<std::string>> TextToPhonemeConverter::create_pronunciation_dict() {
    // Simplified phoneme dictionary (real implementation would use larger dictionary/neural model)
    static std::unordered_map<std::string, std::vector<std::string>> dict = {
        {"hello", {"h", "eh", "l", "ow"}},
        {"world", {"w", "er", "l", "d"}},
        {"the", {"dh", "ah"}},
        {"cat", {"k", "ae", "t"}},
        {"dog", {"d", "ao", "g"}},
        {"house", {"h", "aw", "s"}},
        {"car", {"k", "aa", "r"}},
        {"water", {"w", "ao", "t", "er"}},
        {"computer", {"k", "ah", "m", "p", "y", "uw", "t", "er"}},
        {"speech", {"s", "p", "iy", "ch"}},
        {"synthesis", {"s", "ih", "n", "th", "ah", "s", "ih", "s"}},
        {"artificial", {"aa", "r", "t", "ih", "f", "ih", "sh", "ah", "l"}},
        {"intelligence", {"ih", "n", "t", "eh", "l", "ih", "jh", "ah", "n", "s"}},
        {"learning", {"l", "er", "n", "ih", "ng"}},
        {"machine", {"m", "ah", "sh", "iy", "n"}},
        {"neural", {"n", "uw", "r", "ah", "l"}},
        {"network", {"n", "eh", "t", "w", "er", "k"}},
        {"model", {"m", "aa", "d", "ah", "l"}},
        {"test", {"t", "eh", "s", "t"}},
        {"demo", {"d", "eh", "m", "ow"}},
        {"good", {"g", "uh", "d"}},
        {"morning", {"m", "ao", "r", "n", "ih", "ng"}},
        {"afternoon", {"ae", "f", "t", "er", "n", "uw", "n"}},
        {"evening", {"iy", "v", "n", "ih", "ng"}},
        {"night", {"n", "ay", "t"}},
        {"thank", {"th", "ae", "ng", "k"}},
        {"you", {"y", "uw"}},
        {"please", {"p", "l", "iy", "z"}},
        {"welcome", {"w", "eh", "l", "k", "ah", "m"}}
    };
    return dict;
}

std::string TextToPhonemeConverter::preprocess_text(const std::string& text) {
    std::string processed = text;
    
    // Convert to lowercase
    std::transform(processed.begin(), processed.end(), processed.begin(), ::tolower);
    
    // Remove punctuation except for sentence boundaries
    std::regex punct_regex("[,.!?;:]");
    processed = std::regex_replace(processed, punct_regex, " ");
    
    // Normalize whitespace
    std::regex ws_regex("\\s+");
    processed = std::regex_replace(processed, ws_regex, " ");
    
    // Trim
    processed.erase(0, processed.find_first_not_of(" \t\r\n"));
    processed.erase(processed.find_last_not_of(" \t\r\n") + 1);
    
    return processed;
}

std::vector<std::string> TextToPhonemeConverter::lookup_word_phonemes(const std::string& word) {
    static auto dict = create_pronunciation_dict();
    
    auto it = dict.find(word);
    if (it != dict.end()) {
        return it->second;
    }
    
    // Fallback: simple letter-to-phoneme mapping for unknown words
    std::vector<std::string> phonemes;
    for (char c : word) {
        switch (c) {
            case 'a': phonemes.push_back("ae"); break;
            case 'e': phonemes.push_back("eh"); break;
            case 'i': phonemes.push_back("ih"); break;
            case 'o': phonemes.push_back("aa"); break;
            case 'u': phonemes.push_back("ah"); break;
            case 'b': phonemes.push_back("b"); break;
            case 'c': phonemes.push_back("k"); break;
            case 'd': phonemes.push_back("d"); break;
            case 'f': phonemes.push_back("f"); break;
            case 'g': phonemes.push_back("g"); break;
            case 'h': phonemes.push_back("h"); break;
            case 'j': phonemes.push_back("jh"); break;
            case 'k': phonemes.push_back("k"); break;
            case 'l': phonemes.push_back("l"); break;
            case 'm': phonemes.push_back("m"); break;
            case 'n': phonemes.push_back("n"); break;
            case 'p': phonemes.push_back("p"); break;
            case 'q': phonemes.push_back("k"); break;
            case 'r': phonemes.push_back("r"); break;
            case 's': phonemes.push_back("s"); break;
            case 't': phonemes.push_back("t"); break;
            case 'v': phonemes.push_back("v"); break;
            case 'w': phonemes.push_back("w"); break;
            case 'x': phonemes.push_back("k"); phonemes.push_back("s"); break;
            case 'y': phonemes.push_back("y"); break;
            case 'z': phonemes.push_back("z"); break;
            default: break; // Skip unknown characters
        }
    }
    
    return phonemes;
}

std::vector<Phoneme> TextToPhonemeConverter::text_to_phonemes(const std::string& text) {
    std::string processed = preprocess_text(text);
    std::istringstream iss(processed);
    std::string word;
    std::vector<Phoneme> phonemes;
    
    while (iss >> word) {
        auto word_phonemes = lookup_word_phonemes(word);
        
        for (const auto& phoneme_str : word_phonemes) {
            Phoneme phoneme;
            phoneme.symbol = phoneme_str;
            
            // Assign typical durations (simplified)
            if (phoneme_str == "ae" || phoneme_str == "eh" || phoneme_str == "ih" || 
                phoneme_str == "aa" || phoneme_str == "ah" || phoneme_str == "ow" ||
                phoneme_str == "aw" || phoneme_str == "iy" || phoneme_str == "uw" ||
                phoneme_str == "ay" || phoneme_str == "er" || phoneme_str == "ao") {
                // Vowels - longer duration
                phoneme.duration = 0.12 + (rand() % 40) / 1000.0; // 120-160ms
                phoneme.pitch = 150.0 + (rand() % 100); // Pitch variation
                phoneme.energy = 0.7 + (rand() % 30) / 100.0;
            } else {
                // Consonants - shorter duration  
                phoneme.duration = 0.08 + (rand() % 30) / 1000.0; // 80-110ms
                phoneme.pitch = 120.0 + (rand() % 60);
                phoneme.energy = 0.5 + (rand() % 30) / 100.0;
            }
            
            phonemes.push_back(phoneme);
        }
        
        // Add word boundary (short pause)
        Phoneme pause;
        pause.symbol = "SIL";
        pause.duration = 0.05; // 50ms pause
        pause.pitch = 0.0;
        pause.energy = 0.0;
        phonemes.push_back(pause);
    }
    
    return phonemes;
}

// ===== NeuralVocoder Implementation =====

NeuralVocoder::NeuralVocoder(const VoiceConfig& config) : voice_config_(config) {
}

Tensor NeuralVocoder::generate_carrier_wave(double frequency, double duration, double phase) {
    size_t num_samples = static_cast<size_t>(duration * voice_config_.sample_rate);
    Tensor wave({num_samples});
    
    for (size_t i = 0; i < num_samples; ++i) {
        double t = static_cast<double>(i) / voice_config_.sample_rate;
        wave({i}) = std::sin(2.0 * M_PI * frequency * t + phase);
    }
    
    return wave;
}

Tensor NeuralVocoder::apply_formant_filtering(const Tensor& carrier, const std::string& phoneme) {
    // Simplified formant synthesis (real implementation would use proper filtering)
    Tensor filtered = carrier;
    
    // Apply basic frequency shaping based on phoneme
    double formant_freq = 800.0; // Default formant
    
    if (phoneme == "ae" || phoneme == "aa") formant_freq = 700.0;
    else if (phoneme == "eh" || phoneme == "ah") formant_freq = 500.0;
    else if (phoneme == "ih" || phoneme == "iy") formant_freq = 2200.0;
    else if (phoneme == "ow" || phoneme == "uw") formant_freq = 350.0;
    else if (phoneme == "er") formant_freq = 1400.0;
    
    // Simple amplitude modulation to simulate formants
    for (size_t i = 0; i < filtered.size(); ++i) {
        double t = static_cast<double>(i) / voice_config_.sample_rate;
        double formant_mod = 0.3 * std::sin(2.0 * M_PI * formant_freq * t);
        filtered.data()[i] *= (1.0 + formant_mod);
    }
    
    return filtered;
}

Tensor NeuralVocoder::apply_envelope(const Tensor& audio, double attack, double decay, double sustain, double release) {
    Tensor shaped = audio;
    size_t num_samples = shaped.size();
    
    size_t attack_samples = static_cast<size_t>(attack * voice_config_.sample_rate);
    size_t decay_samples = static_cast<size_t>(decay * voice_config_.sample_rate);
    size_t release_samples = static_cast<size_t>(release * voice_config_.sample_rate);
    
    for (size_t i = 0; i < num_samples; ++i) {
        double envelope = 1.0;
        
        if (i < attack_samples) {
            // Attack phase
            envelope = static_cast<double>(i) / attack_samples;
        } else if (i < attack_samples + decay_samples) {
            // Decay phase
            double decay_progress = static_cast<double>(i - attack_samples) / decay_samples;
            envelope = 1.0 - decay_progress * (1.0 - sustain);
        } else if (i >= num_samples - release_samples) {
            // Release phase
            double release_progress = static_cast<double>(num_samples - i) / release_samples;
            envelope = sustain * release_progress;
        } else {
            // Sustain phase
            envelope = sustain;
        }
        
        shaped.data()[i] *= envelope;
    }
    
    return shaped;
}

Tensor NeuralVocoder::synthesize_from_phonemes(const std::vector<Phoneme>& phonemes) {
    std::vector<Tensor> segments;
    double total_duration = 0.0;
    
    for (const auto& phoneme : phonemes) {
        total_duration += phoneme.duration;
    }
    
    size_t total_samples = static_cast<size_t>(total_duration * voice_config_.sample_rate);
    Tensor audio({total_samples});
    
    size_t current_sample = 0;
    
    for (const auto& phoneme : phonemes) {
        if (phoneme.symbol == "SIL") {
            // Silence - just advance the sample pointer
            size_t silence_samples = static_cast<size_t>(phoneme.duration * voice_config_.sample_rate);
            current_sample += silence_samples;
            continue;
        }
        
        // Generate carrier wave
        Tensor carrier = generate_carrier_wave(phoneme.pitch, phoneme.duration);
        
        // Apply formant filtering
        Tensor formant_filtered = apply_formant_filtering(carrier, phoneme.symbol);
        
        // Apply amplitude envelope
        Tensor shaped = apply_envelope(formant_filtered, 0.01, 0.02, 0.8, 0.03);
        
        // Apply energy scaling
        for (size_t i = 0; i < shaped.size(); ++i) {
            shaped.data()[i] *= phoneme.energy;
        }
        
        // Add to main audio buffer
        for (size_t i = 0; i < shaped.size() && current_sample + i < total_samples; ++i) {
            audio({current_sample + i}) += shaped({i});
        }
        
        current_sample += shaped.size();
    }
    
    // Normalize audio to prevent clipping
    double max_amplitude = 0.0;
    for (size_t i = 0; i < audio.size(); ++i) {
        max_amplitude = std::max(max_amplitude, std::abs(audio.data()[i]));
    }
    
    if (max_amplitude > 0.0) {
        double scale = 0.9 / max_amplitude;
        for (size_t i = 0; i < audio.size(); ++i) {
            audio.data()[i] *= scale;
        }
    }
    
    return audio;
}

Tensor NeuralVocoder::phonemes_to_mel_spectrogram(const std::vector<Phoneme>& phonemes) {
    // Placeholder implementation - convert phonemes to mel spectrogram
    size_t mel_bins = 80;
    size_t time_frames = phonemes.size() * 10; // ~10 frames per phoneme
    
    Tensor mel_spec({mel_bins, time_frames});
    
    for (size_t t = 0; t < time_frames; ++t) {
        size_t phoneme_idx = t / 10;
        if (phoneme_idx >= phonemes.size()) phoneme_idx = phonemes.size() - 1;
        
        const auto& phoneme = phonemes[phoneme_idx];
        
        for (size_t m = 0; m < mel_bins; ++m) {
            // Create spectral characteristics based on phoneme
            double freq = 80.0 + (m * 200.0); // Frequency corresponding to mel bin
            double energy = phoneme.energy;
            
            if (phoneme.symbol == "SIL") {
                mel_spec({m, t}) = -10.0; // Low energy for silence
            } else {
                // Simulate formant structure
                double resonance = std::exp(-std::pow((freq - phoneme.pitch) / 100.0, 2));
                mel_spec({m, t}) = energy * resonance + 0.1 * (rand() / double(RAND_MAX));
            }
        }
    }
    
    return mel_spec;
}

Tensor NeuralVocoder::mel_spectrogram_to_audio(const Tensor& mel_spec) {
    // Placeholder implementation - convert mel spectrogram to audio
    auto shape = mel_spec.shape();
    size_t mel_bins = shape[0];
    size_t time_frames = shape[1];
    
    // Simple inverse: generate audio based on spectral content
    size_t samples_per_frame = 256; // Hop size
    size_t total_samples = time_frames * samples_per_frame;
    
    Tensor audio({total_samples});
    
    for (size_t t = 0; t < time_frames; ++t) {
        for (size_t s = 0; s < samples_per_frame; ++s) {
            double sample = 0.0;
            
            for (size_t m = 0; m < mel_bins; ++m) {
                double freq = 80.0 + (m * 200.0);
                double amplitude = mel_spec({m, t});
                double phase = 2.0 * M_PI * freq * (t * samples_per_frame + s) / voice_config_.sample_rate;
                sample += amplitude * std::sin(phase) / mel_bins;
            }
            
            size_t sample_idx = t * samples_per_frame + s;
            if (sample_idx < total_samples) {
                audio({sample_idx}) = sample * 0.1; // Scale down
            }
        }
    }
    
    return audio;
}

// ===== TextToSpeechPipeline Implementation =====

TextToSpeechPipeline::TextToSpeechPipeline(const VoiceConfig& config) 
    : voice_config_(config), vocoder_(std::make_unique<NeuralVocoder>(config)) {
}

Tensor TextToSpeechPipeline::synthesize(const std::string& text) {
    // Convert text to phonemes
    auto phonemes = TextToPhonemeConverter::text_to_phonemes(text);
    
    // Synthesize audio from phonemes
    return vocoder_->synthesize_from_phonemes(phonemes);
}

std::vector<Tensor> TextToSpeechPipeline::batch_synthesize(const std::vector<std::string>& texts) {
    std::vector<Tensor> results;
    results.reserve(texts.size());
    
    for (const auto& text : texts) {
        results.push_back(synthesize(text));
    }
    
    return results;
}

void TextToSpeechPipeline::set_voice_config(const VoiceConfig& config) {
    voice_config_ = config;
    vocoder_ = std::make_unique<NeuralVocoder>(config);
}

const VoiceConfig& TextToSpeechPipeline::get_voice_config() const {
    return voice_config_;
}

} // namespace ai
} // namespace asekioml
