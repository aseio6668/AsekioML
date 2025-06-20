#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <iostream>
#include <sstream>
#include <algorithm>

namespace clmodel {

// Token representation with debugging info
struct Token {
    std::string text;
    int id;
    size_t start_pos;
    size_t end_pos;
    
    Token(const std::string& t, int i, size_t start = 0, size_t end = 0)
        : text(t), id(i), start_pos(start), end_pos(end) {}
};

// Tokenization result with full transparency
struct TokenizationResult {
    std::vector<Token> tokens;
    std::string original_text;
    size_t total_chars;
    size_t total_tokens;
    bool truncated;
    size_t max_length;
    
    // Debugging helpers
    void print_debug() const {
        std::cout << "=== Tokenization Debug Info ===" << std::endl;
        std::cout << "Original text length: " << total_chars << " chars" << std::endl;
        std::cout << "Total tokens: " << total_tokens << std::endl;
        std::cout << "Truncated: " << (truncated ? "YES" : "NO") << std::endl;
        std::cout << "Max length: " << max_length << std::endl;
        std::cout << std::endl;
        
        std::cout << "Token breakdown:" << std::endl;
        for (size_t i = 0; i < tokens.size(); ++i) {
            const auto& token = tokens[i];
            std::cout << "  [" << i << "] '" << token.text << "' (id:" << token.id 
                     << ", pos:" << token.start_pos << "-" << token.end_pos << ")" << std::endl;
        }
        std::cout << std::endl;
    }
    
    std::string reconstruct() const {
        std::string result;
        for (const auto& token : tokens) {
            result += token.text;
        }
        return result;
    }
    
    bool validate() const {
        std::string reconstructed = reconstruct();
        if (reconstructed != original_text && !truncated) {
            std::cerr << "WARNING: Tokenization validation failed!" << std::endl;
            std::cerr << "Original: '" << original_text << "'" << std::endl;
            std::cerr << "Reconstructed: '" << reconstructed << "'" << std::endl;
            return false;
        }
        return true;
    }
};

// Simple but transparent tokenizer implementation
class Tokenizer {
private:
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> reverse_vocab_;
    int next_id_;
    size_t max_length_;
    
    // Special tokens
    static constexpr int UNK_TOKEN_ID = 0;
    static constexpr int PAD_TOKEN_ID = 1;
    static constexpr int BOS_TOKEN_ID = 2;
    static constexpr int EOS_TOKEN_ID = 3;
    
public:
    Tokenizer(size_t max_length = 512) 
        : next_id_(4), max_length_(max_length) {
        // Initialize special tokens
        vocab_["<UNK>"] = UNK_TOKEN_ID;
        vocab_["<PAD>"] = PAD_TOKEN_ID;
        vocab_["<BOS>"] = BOS_TOKEN_ID;
        vocab_["<EOS>"] = EOS_TOKEN_ID;
        
        reverse_vocab_[UNK_TOKEN_ID] = "<UNK>";
        reverse_vocab_[PAD_TOKEN_ID] = "<PAD>";
        reverse_vocab_[BOS_TOKEN_ID] = "<BOS>";
        reverse_vocab_[EOS_TOKEN_ID] = "<EOS>";
    }
    
    // Train tokenizer on text (simple word-based for demonstration)
    void train(const std::vector<std::string>& texts) {
        std::cout << "Training tokenizer on " << texts.size() << " texts..." << std::endl;
        
        std::unordered_map<std::string, int> word_counts;
        
        // Count word frequencies
        for (const auto& text : texts) {
            auto words = split_words(text);
            for (const auto& word : words) {
                word_counts[word]++;
            }
        }
        
        // Add words to vocabulary (simple frequency-based)
        std::cout << "Building vocabulary..." << std::endl;
        for (const auto& [word, count] : word_counts) {
            if (count >= 2 && vocab_.find(word) == vocab_.end()) {  // Minimum frequency
                vocab_[word] = next_id_;
                reverse_vocab_[next_id_] = word;
                next_id_++;
            }
        }
        
        std::cout << "Vocabulary size: " << vocab_.size() << " tokens" << std::endl;
    }
    
    // Tokenize text with full transparency
    TokenizationResult tokenize(const std::string& text, bool add_special_tokens = true) const {
        TokenizationResult result;
        result.original_text = text;
        result.total_chars = text.length();
        result.max_length = max_length_;
        result.truncated = false;
        
        std::vector<std::string> words = split_words(text);
        size_t char_pos = 0;
        
        // Add BOS token if requested
        if (add_special_tokens) {
            result.tokens.emplace_back("<BOS>", BOS_TOKEN_ID, 0, 0);
        }
        
        // Tokenize words
        for (const auto& word : words) {
            // Find word position in original text
            size_t word_start = text.find(word, char_pos);
            size_t word_end = word_start + word.length();
            
            int token_id = vocab_.count(word) ? vocab_.at(word) : UNK_TOKEN_ID;
            std::string token_text = vocab_.count(word) ? word : "<UNK>";
            
            result.tokens.emplace_back(token_text, token_id, word_start, word_end);
            char_pos = word_end;
            
            // Check for truncation
            if (result.tokens.size() >= max_length_ - (add_special_tokens ? 1 : 0)) {
                result.truncated = true;
                break;
            }
        }
        
        // Add EOS token if requested
        if (add_special_tokens) {
            result.tokens.emplace_back("<EOS>", EOS_TOKEN_ID, text.length(), text.length());
        }
        
        result.total_tokens = result.tokens.size();
        return result;
    }
    
    // Convert token IDs back to text
    std::string decode(const std::vector<int>& token_ids) const {
        std::string result;
        for (int id : token_ids) {
            if (reverse_vocab_.count(id)) {
                std::string token = reverse_vocab_.at(id);
                if (token != "<BOS>" && token != "<EOS>" && token != "<PAD>") {
                    if (!result.empty() && token != "<UNK>") {
                        result += " ";
                    }
                    if (token != "<UNK>") {
                        result += token;
                    }
                }
            }
        }
        return result;
    }
    
    // Extract token IDs for model input
    std::vector<int> get_ids(const TokenizationResult& result) const {
        std::vector<int> ids;
        ids.reserve(result.tokens.size());
        for (const auto& token : result.tokens) {
            ids.push_back(token.id);
        }
        return ids;
    }
    
    // Vocabulary introspection
    void print_vocab_stats() const {
        std::cout << "=== Tokenizer Vocabulary Stats ===" << std::endl;
        std::cout << "Total vocabulary size: " << vocab_.size() << std::endl;
        std::cout << "Special tokens: 4 (<UNK>, <PAD>, <BOS>, <EOS>)" << std::endl;
        std::cout << "Regular tokens: " << (vocab_.size() - 4) << std::endl;
        std::cout << "Max sequence length: " << max_length_ << std::endl;
        std::cout << std::endl;
    }
    
    size_t vocab_size() const { return vocab_.size(); }
    size_t max_length() const { return max_length_; }
    
    // Debugging: check if text would be truncated
    bool would_truncate(const std::string& text) const {
        auto words = split_words(text);
        return words.size() + 2 > max_length_;  // +2 for BOS/EOS
    }

private:
    std::vector<std::string> split_words(const std::string& text) const {
        std::vector<std::string> words;
        std::istringstream iss(text);
        std::string word;
        
        while (iss >> word) {
            // Simple preprocessing: lowercase and remove punctuation
            std::string clean_word;            for (char c : word) {
                if (std::isalnum(static_cast<unsigned char>(c))) {
                    clean_word += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                }
            }
            if (!clean_word.empty()) {
                words.push_back(clean_word);
            }
        }
        
        return words;
    }
};

// Tokenizer factory for different types
class TokenizerFactory {
public:
    enum class Type {
        WORD_BASED,
        CHAR_BASED,
        BPE  // Future implementation
    };
    
    static std::unique_ptr<Tokenizer> create(Type type, size_t max_length = 512) {
        switch (type) {
            case Type::WORD_BASED:
                return std::make_unique<Tokenizer>(max_length);
            case Type::CHAR_BASED:
                // TODO: Implement character-based tokenizer
                throw std::runtime_error("Character-based tokenizer not implemented yet");
            case Type::BPE:
                // TODO: Implement BPE tokenizer
                throw std::runtime_error("BPE tokenizer not implemented yet");
            default:
                throw std::runtime_error("Unknown tokenizer type");
        }
    }
};

} // namespace clmodel
