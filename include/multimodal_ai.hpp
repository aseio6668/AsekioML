#pragma once

#include "network.hpp"
#include "advanced_ai_layers.hpp"
#include <map>
#include <string>
#include <vector>
#include <memory>

namespace asekioml {
namespace ai {

/**
 * @brief Multi-modal input types for different AI applications
 */
enum class ModalityType {
    TEXT,           // Text sequences (tokens)
    IMAGE,          // 2D images (RGB/grayscale)
    AUDIO,          // 1D audio waveforms or spectrograms
    VIDEO,          // 3D video sequences (temporal + spatial)
    EMBEDDING       // Pre-computed embeddings
};

/**
 * @brief Data structure for multi-modal inputs
 */
struct ModalityData {
    ModalityType type;
    Matrix data;
    std::map<std::string, double> metadata;  // Sample rate, dimensions, etc.
    
    ModalityData(ModalityType t, const Matrix& d) : type(t), data(d) {}
};

/**
 * @brief Multi-modal neural network for complex AI tasks
 */
class MultiModalNetwork {
private:
    // Separate encoders for each modality
    std::map<ModalityType, std::unique_ptr<NeuralNetwork>> encoders_;
    
    // Cross-modal fusion network
    std::unique_ptr<NeuralNetwork> fusion_network_;
    
    // Decoders for different output modalities
    std::map<ModalityType, std::unique_ptr<NeuralNetwork>> decoders_;
    
    // Shared latent space dimension
    size_t latent_dim_;
    
    // Training configuration
    bool is_compiled_;
    std::string loss_function_;
    std::string optimizer_;
    double learning_rate_;
    
public:
    MultiModalNetwork(size_t latent_dim = 512);
    ~MultiModalNetwork() = default;
    
    // Encoder management
    void add_text_encoder(size_t vocab_size, size_t max_sequence_length, 
                         size_t d_model = 512, size_t num_heads = 8, size_t num_layers = 6);
    void add_image_encoder(size_t input_channels, size_t input_height, size_t input_width);
    void add_audio_encoder(size_t input_length, size_t sample_rate = 22050);
    void add_video_encoder(size_t input_channels, size_t frames, size_t height, size_t width);
    
    // Decoder management
    void add_text_decoder(size_t vocab_size, size_t max_sequence_length,
                         size_t d_model = 512, size_t num_heads = 8, size_t num_layers = 6);
    void add_image_decoder(size_t output_channels, size_t output_height, size_t output_width);
    void add_audio_decoder(size_t output_length, size_t sample_rate = 22050);
    void add_video_decoder(size_t output_channels, size_t frames, size_t height, size_t width);
    
    // Fusion network configuration
    void configure_fusion(const std::vector<ModalityType>& input_modalities,
                         const std::vector<ModalityType>& output_modalities);
    
    // Compilation
    void compile(const std::string& loss_function, const std::string& optimizer, double learning_rate);
    
    // Forward pass
    std::map<ModalityType, Matrix> predict(const std::map<ModalityType, ModalityData>& inputs);
    
    // Training
    void train_step(const std::map<ModalityType, ModalityData>& inputs,
                   const std::map<ModalityType, ModalityData>& targets);
    
    // Specialized methods for common AI tasks
    Matrix text_to_image(const Matrix& text_tokens, size_t image_height, size_t image_width);
    Matrix text_to_audio(const Matrix& text_tokens, size_t audio_length, double duration_seconds = 10.0);
    Matrix text_to_video(const Matrix& text_tokens, size_t frames, size_t height, size_t width);
    Matrix image_to_text(const Matrix& image, size_t max_text_length);
    Matrix audio_to_text(const Matrix& audio, size_t max_text_length);
    
    // Advanced generation methods
    std::vector<Matrix> generate_movie_sequence(const Matrix& script_tokens, 
                                              const Matrix& style_image,
                                              const Matrix& background_audio,
                                              size_t num_frames, size_t frame_height, size_t frame_width);
    
    Matrix generate_orchestral_score(const Matrix& text_description, 
                                    const std::vector<std::string>& instruments,
                                    double duration_seconds);
    
    // Utility methods
    void set_training_mode(bool training);
    bool is_compiled() const { return is_compiled_; }
    size_t count_parameters() const;
    
    // Serialization
    bool save(const std::string& filepath) const;
    bool load(const std::string& filepath);
    
private:
    void initialize_fusion_network(const std::vector<ModalityType>& input_modalities,
                                  const std::vector<ModalityType>& output_modalities);
    Matrix encode_modality(ModalityType type, const Matrix& data);
    Matrix decode_to_modality(ModalityType type, const Matrix& latent_representation);
    Matrix apply_cross_modal_attention(const std::vector<Matrix>& encoded_modalities);
};

/**
 * @brief Specialized network architectures for specific AI tasks
 */
class TextToImageNetwork : public MultiModalNetwork {
public:
    TextToImageNetwork(size_t vocab_size, size_t max_text_length, 
                      size_t image_height, size_t image_width, size_t image_channels = 3);
    
    Matrix generate_image(const Matrix& text_tokens, 
                         const Matrix& style_embedding = Matrix(0, 0),
                         double guidance_scale = 7.5);
    
    void train_on_text_image_pairs(const std::vector<Matrix>& text_samples,
                                  const std::vector<Matrix>& image_samples,
                                  int epochs = 100);
};

class TextToAudioNetwork : public MultiModalNetwork {
public:
    TextToAudioNetwork(size_t vocab_size, size_t max_text_length,
                      size_t audio_length, size_t sample_rate = 22050);
    
    Matrix generate_speech(const Matrix& text_tokens, 
                          const Matrix& speaker_embedding = Matrix(0, 0));
    
    Matrix generate_music(const Matrix& text_description,
                         const std::string& genre = "classical",
                         double tempo_bpm = 120.0);
};

class TextToVideoNetwork : public MultiModalNetwork {
public:
    TextToVideoNetwork(size_t vocab_size, size_t max_text_length,
                      size_t num_frames, size_t frame_height, size_t frame_width,
                      size_t frame_channels = 3);
    
    std::vector<Matrix> generate_video(const Matrix& text_tokens,
                                     const Matrix& style_reference = Matrix(0, 0),
                                     double fps = 24.0);
    
    std::vector<Matrix> generate_movie_trailer(const Matrix& script_tokens,
                                             const Matrix& genre_embedding,
                                             const Matrix& character_descriptions);
};

/**
 * @brief Orchestral multi-model system for complete content creation
 */
class OrchestralAISystem {
private:
    std::unique_ptr<TextToImageNetwork> text_to_image_;
    std::unique_ptr<TextToAudioNetwork> text_to_audio_;
    std::unique_ptr<TextToVideoNetwork> text_to_video_;
    std::unique_ptr<MultiModalNetwork> orchestrator_;
    
    // Content pipeline management
    struct ContentPipeline {
        std::vector<std::string> stages;
        std::map<std::string, ModalityType> stage_outputs;
        std::map<std::string, std::function<Matrix(const Matrix&)>> processors;
    };
    
    ContentPipeline current_pipeline_;
    
public:
    OrchestralAISystem();
    
    // High-level content generation
    struct MovieProject {
        Matrix script;
        Matrix style_guide;
        Matrix character_descriptions;
        Matrix music_theme;
        std::vector<std::string> scene_descriptions;
    };
    
    struct GeneratedMovie {
        std::vector<Matrix> video_frames;
        Matrix audio_track;
        Matrix synchronized_subtitles;
        std::map<std::string, Matrix> character_voices;
    };
    
    GeneratedMovie create_full_movie(const MovieProject& project,
                                   double duration_minutes = 5.0,
                                   double fps = 24.0);
    
    // Live streaming content generation
    class LiveStreamGenerator {
    private:
        OrchestralAISystem* parent_;
        bool is_streaming_;
        
    public:
        LiveStreamGenerator(OrchestralAISystem* parent) : parent_(parent), is_streaming_(false) {}
        
        void start_stream(const Matrix& initial_prompt, 
                         const std::map<std::string, double>& stream_config);
        void update_stream_content(const Matrix& new_prompt_delta);
        Matrix get_next_frame();
        Matrix get_current_audio_chunk();
        void stop_stream();
    };
    
    std::unique_ptr<LiveStreamGenerator> create_live_stream();
    
    // Content style transfer and adaptation
    Matrix adapt_content_style(const Matrix& source_content, 
                              const Matrix& target_style,
                              ModalityType content_type);
    
    // Training and fine-tuning
    void train_on_content_dataset(const std::string& dataset_path, 
                                 const std::vector<ModalityType>& modalities,
                                 int epochs = 50);
    
    // System management
    void save_system_state(const std::string& checkpoint_path);
    void load_system_state(const std::string& checkpoint_path);
    void optimize_for_realtime();
    
private:
    void initialize_specialized_networks();
    void setup_content_pipeline();
    Matrix synchronize_audio_video(const Matrix& video_frames, const Matrix& audio);
    std::vector<Matrix> apply_temporal_consistency(const std::vector<Matrix>& frames);
};

} // namespace ai
} // namespace asekioml
