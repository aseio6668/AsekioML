#pragma once

#include "../tensor.hpp"
#include "audio_processing.hpp"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace clmodel {
namespace ai {

/**
 * @brief Phoneme representation for TTS
 */
struct Phoneme {
    std::string symbol;
    double duration;  // in seconds
    double pitch;     // in Hz
    double energy;    // amplitude scale
};

/**
 * @brief Voice characteristics for TTS synthesis
 */
struct VoiceConfig {
    double base_pitch = 150.0;     // Base fundamental frequency (Hz)
    double pitch_range = 50.0;     // Pitch variation range
    double speaking_rate = 1.0;    // Speed multiplier
    double voice_quality = 1.0;    // Voice quality factor
    int sample_rate = 22050;       // Output sample rate
};

/**
 * @brief Text-to-Phoneme converter for pronunciation
 */
class TextToPhonemeConverter {
public:
    /**
     * @brief Convert text to phoneme sequence
     * @param text Input text to convert
     * @return Vector of phonemes with timing
     */
    static std::vector<Phoneme> text_to_phonemes(const std::string& text);
    
    /**
     * @brief Process text for TTS (normalization, expansion)
     * @param text Raw input text
     * @return Processed text ready for phoneme conversion
     */
    static std::string preprocess_text(const std::string& text);
    
private:
    static std::unordered_map<std::string, std::vector<std::string>> create_pronunciation_dict();
    static std::vector<std::string> lookup_word_phonemes(const std::string& word);
};

/**
 * @brief Neural vocoder for converting linguistic features to audio
 */
class NeuralVocoder {
public:
    /**
     * @brief Initialize vocoder with voice configuration
     */
    NeuralVocoder(const VoiceConfig& config = VoiceConfig{});
    
    /**
     * @brief Synthesize audio from phoneme sequence
     * @param phonemes Input phoneme sequence
     * @return Audio tensor [samples] or [samples, channels]
     */
    Tensor synthesize_from_phonemes(const std::vector<Phoneme>& phonemes);
    
    /**
     * @brief Generate mel spectrogram from phonemes
     * @param phonemes Input phoneme sequence
     * @return Mel spectrogram tensor [mel_bins, time_frames]
     */
    Tensor phonemes_to_mel_spectrogram(const std::vector<Phoneme>& phonemes);
    
    /**
     * @brief Convert mel spectrogram to audio waveform
     * @param mel_spec Input mel spectrogram
     * @return Audio waveform tensor
     */
    Tensor mel_spectrogram_to_audio(const Tensor& mel_spec);
    
private:
    VoiceConfig voice_config_;
    
    // Helper methods for audio synthesis
    Tensor generate_carrier_wave(double frequency, double duration, double phase = 0.0);
    Tensor apply_formant_filtering(const Tensor& carrier, const std::string& phoneme);
    Tensor apply_envelope(const Tensor& audio, double attack, double decay, double sustain, double release);
};

/**
 * @brief Complete Text-to-Speech pipeline
 */
class TextToSpeechPipeline {
public:
    /**
     * @brief Initialize TTS pipeline with voice configuration
     */
    TextToSpeechPipeline(const VoiceConfig& config = VoiceConfig{});
    
    /**
     * @brief Convert text to speech audio
     * @param text Input text to synthesize
     * @return Audio tensor containing synthesized speech
     */
    Tensor synthesize(const std::string& text);
    
    /**
     * @brief Batch text-to-speech conversion
     * @param texts Vector of input texts
     * @return Vector of audio tensors
     */
    std::vector<Tensor> batch_synthesize(const std::vector<std::string>& texts);
    
    /**
     * @brief Set voice characteristics
     */
    void set_voice_config(const VoiceConfig& config);
    
    /**
     * @brief Get current voice configuration
     */
    const VoiceConfig& get_voice_config() const;
    
private:
    VoiceConfig voice_config_;
    std::unique_ptr<NeuralVocoder> vocoder_;
};

} // namespace ai
} // namespace clmodel
