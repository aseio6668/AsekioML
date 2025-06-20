#include "../../include/ai/text_to_image.hpp"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <random>
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace asekioml {
namespace ai {

// ===== TextProcessor Implementation =====

std::vector<std::string> TextProcessor::tokenize_words(const std::string& text) {
    std::vector<std::string> tokens;
    std::string cleaned = clean_text(text);
    std::istringstream iss(cleaned);
    std::string word;
    
    while (iss >> word) {
        if (!word.empty()) {
            tokens.push_back(word);
        }
    }
    
    return tokens;
}

std::vector<std::string> TextProcessor::tokenize_chars(const std::string& text) {
    std::vector<std::string> tokens;
    for (char c : text) {
        if (!is_whitespace(c)) {
            tokens.push_back(std::string(1, c));
        }
    }
    return tokens;
}

std::vector<int> TextProcessor::encode_tokens(const std::vector<std::string>& tokens,
                                            const std::unordered_map<std::string, int>& vocab,
                                            int unk_token) {
    std::vector<int> token_ids;
    token_ids.reserve(tokens.size());
    
    for (const auto& token : tokens) {
        auto it = vocab.find(token);
        if (it != vocab.end()) {
            token_ids.push_back(it->second);
        } else {
            token_ids.push_back(unk_token);
        }
    }
    
    return token_ids;
}

std::unordered_map<std::string, int> TextProcessor::build_vocabulary(
    const std::vector<std::string>& texts,
    size_t max_vocab_size,
    int min_freq) {
    
    // Count token frequencies
    std::unordered_map<std::string, int> token_counts;
    
    for (const auto& text : texts) {
        auto tokens = tokenize_words(text);
        for (const auto& token : tokens) {
            token_counts[token]++;
        }
    }
    
    // Sort tokens by frequency
    std::vector<std::pair<std::string, int>> sorted_tokens(token_counts.begin(), token_counts.end());
    std::sort(sorted_tokens.begin(), sorted_tokens.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Build vocabulary
    std::unordered_map<std::string, int> vocab;
    vocab["<UNK>"] = 0;  // Unknown token
    vocab["<PAD>"] = 1;  // Padding token
    vocab["<BOS>"] = 2;  // Beginning of sequence
    vocab["<EOS>"] = 3;  // End of sequence
    
    int token_id = 4;
    for (const auto& [token, count] : sorted_tokens) {
        if (count >= min_freq && vocab.size() < max_vocab_size) {
            vocab[token] = token_id++;
        }
    }
    
    return vocab;
}

Tensor TextProcessor::tokens_to_embeddings(const std::vector<int>& token_ids,
                                         size_t embedding_dim,
                                         size_t vocab_size,
                                         size_t max_length) {
    // Create embedding tensor
    Tensor embeddings({max_length, embedding_dim});
    
    // Simple embedding: use token ID to generate deterministic embedding
    for (size_t i = 0; i < max_length; ++i) {
        int token_id = (i < token_ids.size()) ? token_ids[i] : 1; // Use PAD token for padding
        
        for (size_t j = 0; j < embedding_dim; ++j) {
            // Simple embedding: sin/cos encoding based on token ID and position
            double value = std::sin(token_id * 0.1 + j * 0.01) * 0.1;
            embeddings({i, j}) = value;
        }
    }
    
    return embeddings;
}

Tensor TextProcessor::positional_encoding(size_t seq_length, size_t embedding_dim) {
    Tensor pos_encoding({seq_length, embedding_dim});
    
    for (size_t pos = 0; pos < seq_length; ++pos) {
        for (size_t i = 0; i < embedding_dim; i += 2) {
            double div_term = std::exp(i * (-std::log(10000.0) / embedding_dim));
            
            if (i < embedding_dim) {
                pos_encoding({pos, i}) = std::sin(pos * div_term);
            }
            if (i + 1 < embedding_dim) {
                pos_encoding({pos, i + 1}) = std::cos(pos * div_term);
            }
        }
    }
    
    return pos_encoding;
}

std::string TextProcessor::clean_text(const std::string& text) {
    std::string cleaned = to_lowercase(text);
    cleaned = remove_punctuation(cleaned);
    
    // Remove extra whitespace
    std::string result;
    bool prev_space = false;
    for (char c : cleaned) {
        if (is_whitespace(c)) {
            if (!prev_space && !result.empty()) {
                result += ' ';
                prev_space = true;
            }
        } else {
            result += c;
            prev_space = false;
        }
    }
    
    return result;
}

std::string TextProcessor::to_lowercase(const std::string& text) {
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return lower;
}

std::string TextProcessor::remove_punctuation(const std::string& text) {
    std::string result;
    for (char c : text) {
        if (!is_punctuation(c)) {
            result += c;
        }
    }
    return result;
}

std::vector<std::string> TextProcessor::split_string(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    
    return tokens;
}

bool TextProcessor::is_punctuation(char c) {
    return std::ispunct(static_cast<unsigned char>(c));
}

bool TextProcessor::is_whitespace(char c) {
    return std::isspace(static_cast<unsigned char>(c));
}

// ===== DiffusionModel Implementation =====

std::vector<double> DiffusionModel::linear_schedule(int num_steps, double beta_start, double beta_end) {
    std::vector<double> betas(num_steps);
    double step_size = (beta_end - beta_start) / (num_steps - 1);
    
    for (int i = 0; i < num_steps; ++i) {
        betas[i] = beta_start + i * step_size;
    }
    
    return betas;
}

std::vector<double> DiffusionModel::cosine_schedule(int num_steps, double s) {
    std::vector<double> betas(num_steps);
    
    auto f = [s](double t) {
        return std::cos((t + s) / (1 + s) * M_PI / 2);
    };
    
    std::vector<double> alpha_cumprod(num_steps + 1);
    for (int i = 0; i <= num_steps; ++i) {
        double t = static_cast<double>(i) / num_steps;
        alpha_cumprod[i] = f(t) * f(t);
    }
    
    for (int i = 0; i < num_steps; ++i) {
        betas[i] = std::min(1.0 - alpha_cumprod[i + 1] / alpha_cumprod[i], 0.999);
    }
    
    return betas;
}

std::pair<std::vector<double>, std::vector<double>> 
DiffusionModel::compute_alphas(const std::vector<double>& betas) {
    std::vector<double> alphas(betas.size());
    std::vector<double> alpha_cumprod(betas.size());
    
    double cumprod = 1.0;
    for (size_t i = 0; i < betas.size(); ++i) {
        alphas[i] = 1.0 - betas[i];
        cumprod *= alphas[i];
        alpha_cumprod[i] = cumprod;
    }
    
    return {alphas, alpha_cumprod};
}

Tensor DiffusionModel::add_noise(const Tensor& x0, int t, const std::vector<double>& alpha_cumprod) {
    if (t >= static_cast<int>(alpha_cumprod.size())) {
        throw std::invalid_argument("Timestep t exceeds schedule length");
    }
    
    double sqrt_alpha_cumprod = std::sqrt(alpha_cumprod[t]);
    double sqrt_one_minus_alpha_cumprod = std::sqrt(1.0 - alpha_cumprod[t]);
    
    auto noise = sample_noise(x0.shape());
    
    // x_t = sqrt(alpha_cumprod_t) * x0 + sqrt(1 - alpha_cumprod_t) * noise
    return x0 * sqrt_alpha_cumprod + noise * sqrt_one_minus_alpha_cumprod;
}

Tensor DiffusionModel::sample_noise(const std::vector<size_t>& shape) {
    Tensor noise(shape);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    
    // Fill with random noise
    size_t total_size = 1;
    for (size_t dim : shape) {
        total_size *= dim;
    }
    
    for (size_t i = 0; i < total_size; ++i) {
        std::vector<size_t> indices;
        size_t temp = i;
        for (int j = shape.size() - 1; j >= 0; --j) {
            indices.insert(indices.begin(), temp % shape[j]);
            temp /= shape[j];
        }
        noise(indices) = dist(gen);
    }
    
    return noise;
}

Tensor DiffusionModel::denoise_step(const Tensor& x_t,
                                  const Tensor& predicted_noise,
                                  int t,
                                  const std::vector<double>& alphas,
                                  const std::vector<double>& alpha_cumprod,
                                  const std::vector<double>& betas) {
    if (t < 0 || t >= static_cast<int>(alphas.size())) {
        throw std::invalid_argument("Invalid timestep");
    }
    
    double alpha_t = alphas[t];
    double alpha_cumprod_t = alpha_cumprod[t];
    double beta_t = betas[t];
    
    double alpha_cumprod_prev = (t > 0) ? alpha_cumprod[t - 1] : 1.0;
    
    // Compute coefficients
    double pred_x0_coeff = 1.0 / std::sqrt(alpha_t);
    double noise_coeff = beta_t / std::sqrt(1.0 - alpha_cumprod_t);
    
    // Predict x0
    auto pred_x0 = (x_t - predicted_noise * noise_coeff) * pred_x0_coeff;
    
    // Compute mean of posterior
    double dir_xt_coeff = std::sqrt(alpha_cumprod_prev) * beta_t / (1.0 - alpha_cumprod_t);
    double xt_coeff = std::sqrt(alpha_t) * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t);
    
    auto mean = pred_x0 * dir_xt_coeff + x_t * xt_coeff;
    
    if (t > 0) {
        // Add noise for non-final step
        double variance = (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t) * beta_t;
        auto noise = sample_noise(x_t.shape());
        return mean + noise * std::sqrt(variance);
    } else {
        return mean;
    }
}

Tensor DiffusionModel::sample_loop(const Tensor& x_t,
                                 const Tensor& text_embeddings,
                                 int num_steps,
                                 double guidance_scale) {
    // Create noise schedule
    auto betas = linear_schedule(num_steps);
    auto [alphas, alpha_cumprod] = compute_alphas(betas);
    
    Tensor x = x_t;
    
    for (int t = num_steps - 1; t >= 0; --t) {
        // Predict noise using simplified U-Net
        auto predicted_noise = simple_unet(x, t, text_embeddings);
        
        // Apply classifier-free guidance (simplified)
        if (guidance_scale > 1.0) {
            // In a real implementation, this would involve conditional and unconditional predictions
            predicted_noise = predicted_noise * guidance_scale;
        }
        
        // Denoise step
        x = denoise_step(x, predicted_noise, t, alphas, alpha_cumprod, betas);
        
        // Progress indicator
        if (t % 10 == 0) {
            std::cout << "Denoising step " << t << "/" << num_steps << std::endl;
        }
    }
    
    return x;
}

Tensor DiffusionModel::simple_unet(const Tensor& x_t, int t, const Tensor& text_embeddings) {
    // This is a greatly simplified placeholder for a U-Net architecture
    // In a real implementation, this would be a complex neural network
    
    auto shape = x_t.shape();
    Tensor predicted_noise(shape);
    
    // Simple noise prediction based on input and timestep
    // This is NOT a real denoising model - just a placeholder
    double time_factor = std::sin(t * 0.1) * 0.1;
    
    size_t total_size = 1;
    for (size_t dim : shape) {
        total_size *= dim;
    }
    
    for (size_t i = 0; i < total_size; ++i) {
        std::vector<size_t> indices;
        size_t temp = i;
        for (int j = shape.size() - 1; j >= 0; --j) {
            indices.insert(indices.begin(), temp % shape[j]);
            temp /= shape[j];
        }
        
        // Simplified noise prediction
        double x_val = x_t(indices);
        double noise_pred = x_val * 0.5 + time_factor;
        
        // Add some influence from text embeddings (very simplified)
        if (!text_embeddings.is_empty() && text_embeddings.size() > 0) {
            double text_influence = 0.0;
            auto text_shape = text_embeddings.shape();
            if (text_shape.size() >= 2 && text_shape[0] > 0 && text_shape[1] > 0) {
                text_influence = text_embeddings({0, 0}) * 0.01;
            }
            noise_pred += text_influence;
        }
        
        predicted_noise(indices) = noise_pred;
    }
    
    return predicted_noise;
}

Tensor DiffusionModel::simple_attention(const Tensor& query, const Tensor& key, const Tensor& value) {
    // Simplified attention mechanism placeholder
    // In a real implementation, this would be proper multi-head attention
    
    auto q_shape = query.shape();
    if (q_shape.size() != 2) {
        throw std::invalid_argument("Query must be 2D tensor");
    }
    
    Tensor output(q_shape);
    
    // Simple weighted combination (not real attention)
    for (size_t i = 0; i < q_shape[0]; ++i) {
        for (size_t j = 0; j < q_shape[1]; ++j) {
            double sum = query({i, j}) * 0.6;
            if (!key.is_empty() && key.shape().size() >= 2 && 
                i < key.shape()[0] && j < key.shape()[1]) {
                sum += key({i, j}) * 0.3;
            }
            if (!value.is_empty() && value.shape().size() >= 2 && 
                i < value.shape()[0] && j < value.shape()[1]) {
                sum += value({i, j}) * 0.1;
            }
            output({i, j}) = sum;
        }
    }
    
    return output;
}

// ===== TextToImagePipeline Implementation =====

TextToImagePipeline::TextToImagePipeline(size_t image_size, int num_steps)
    : image_size_(image_size), num_steps_(num_steps) {
    
    // Initialize noise schedule
    betas_ = DiffusionModel::linear_schedule(num_steps);
    auto [alphas, alpha_cumprod] = DiffusionModel::compute_alphas(betas_);
    alphas_ = alphas;
    alpha_cumprod_ = alpha_cumprod;
    
    // Create simple default vocabulary
    vocabulary_["<UNK>"] = 0;
    vocabulary_["<PAD>"] = 1;
    vocabulary_["<BOS>"] = 2;
    vocabulary_["<EOS>"] = 3;
    vocabulary_["a"] = 4;
    vocabulary_["cat"] = 5;
    vocabulary_["dog"] = 6;
    vocabulary_["house"] = 7;
    vocabulary_["car"] = 8;
    vocabulary_["tree"] = 9;
    vocabulary_["blue"] = 10;
    vocabulary_["red"] = 11;
    vocabulary_["green"] = 12;
    vocabulary_["big"] = 13;
    vocabulary_["small"] = 14;
}

Tensor TextToImagePipeline::generate(const std::string& prompt, double guidance_scale, int seed) {
    if (seed >= 0) {
        std::srand(seed);
    }
    
    std::cout << "Generating image for prompt: \"" << prompt << "\"" << std::endl;
    
    // Process text
    auto text_embeddings = process_text(prompt);
    
    // Generate initial noise
    auto initial_noise = generate_initial_noise();
    
    // Run diffusion sampling
    auto generated_image = DiffusionModel::sample_loop(initial_noise, text_embeddings, 
                                                      num_steps_, guidance_scale);
    
    std::cout << "Image generation complete!" << std::endl;
    return generated_image;
}

std::vector<Tensor> TextToImagePipeline::generate_batch(const std::vector<std::string>& prompts,
                                                       double guidance_scale,
                                                       int seed) {
    std::vector<Tensor> results;
    results.reserve(prompts.size());
    
    for (size_t i = 0; i < prompts.size(); ++i) {
        int batch_seed = (seed >= 0) ? seed + static_cast<int>(i) : -1;
        results.push_back(generate(prompts[i], guidance_scale, batch_seed));
    }
    
    return results;
}

void TextToImagePipeline::set_vocabulary(const std::unordered_map<std::string, int>& vocab) {
    vocabulary_ = vocab;
}

void TextToImagePipeline::load_embeddings(const std::string& embedding_file) {
    // Placeholder for loading pretrained embeddings
    std::cout << "Loading embeddings from: " << embedding_file << " (placeholder)" << std::endl;
}

Tensor TextToImagePipeline::process_text(const std::string& prompt) {
    // Tokenize text
    auto tokens = TextProcessor::tokenize_words(prompt);
    
    // Encode tokens
    auto token_ids = TextProcessor::encode_tokens(tokens, vocabulary_, 0);
    
    // Convert to embeddings
    auto embeddings = TextProcessor::tokens_to_embeddings(token_ids, 512, vocabulary_.size(), 77);
    
    // Add positional encoding
    auto pos_encoding = TextProcessor::positional_encoding(77, 512);
    
    // Combine embeddings and positional encoding
    auto combined = embeddings + pos_encoding;
    
    return combined;
}

Tensor TextToImagePipeline::generate_initial_noise() {
    // Generate initial noise tensor for image [channels, height, width]
    return DiffusionModel::sample_noise({3, image_size_, image_size_});
}

} // namespace ai
} // namespace asekioml
