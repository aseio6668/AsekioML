#include "api/rest_api_server.hpp"
#include "tensor.hpp"
#include "matrix.hpp"
#include "ai/orchestral_ai_director.hpp"
#include "ai/dynamic_model_dispatcher.hpp"
#include "ai/real_time_content_pipeline.hpp"
#include "ai/adaptive_quality_engine.hpp"
#include "ai/production_streaming_manager.hpp"

#include <winsock2.h>
#include <ws2tcpip.h>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>
#include <memory>

#pragma comment(lib, "ws2_32.lib")

#include <iomanip>
#include <chrono>
#include <random>
#include <algorithm>
#include <regex>
#include <stack>

namespace clmodel {

// Convenience aliases
using HttpRequest = api::HttpRequest;
using HttpResponse = api::HttpResponse;
using HttpMethod = api::HttpMethod;
using HttpStatus = api::HttpStatus;
using ApiEndpoint = api::ApiEndpoint;

// AuthToken implementation
double AuthToken::get_current_time() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    return static_cast<double>(time_t);
}

// RequestMetrics implementation
void RequestMetrics::record_request(const std::string& endpoint, double response_time, bool success) {
    total_requests.fetch_add(1);
    if (success) {
        successful_requests.fetch_add(1);
    } else {
        failed_requests.fetch_add(1);
    }
    
    // Update average response time with exponential moving average
    double current_avg = average_response_time.load();
    double new_avg = current_avg * 0.9 + response_time * 0.1;
    average_response_time.store(new_avg);
    
    // Update endpoint-specific counts
    std::lock_guard<std::mutex> lock(metrics_mutex);
    endpoint_counts[endpoint].fetch_add(1);
}

void RequestMetrics::get_summary(std::map<std::string, double>& summary) const {
    summary["total_requests"] = static_cast<double>(total_requests.load());
    summary["successful_requests"] = static_cast<double>(successful_requests.load());
    summary["failed_requests"] = static_cast<double>(failed_requests.load());
    summary["average_response_time"] = average_response_time.load();
    summary["active_connections"] = static_cast<double>(active_connections.load());
    
    if (total_requests.load() > 0) {
        summary["success_rate"] = static_cast<double>(successful_requests.load()) / 
                                 static_cast<double>(total_requests.load());
    } else {
        summary["success_rate"] = 0.0;
    }
}

// RestApiServer implementation
RestApiServer::RestApiServer(const Config& config) 
    : config_(config), metrics_(std::make_unique<RequestMetrics>()), 
      server_running_(false) {
    
#ifdef _WIN32
    server_socket_ = INVALID_SOCKET;
    // Initialize Winsock
    WSADATA wsaData;
    int result = WSAStartup(MAKEWORD(2,2), &wsaData);
    if (result != 0) {
        throw std::runtime_error("WSAStartup failed: " + std::to_string(result));
    }
#else
    server_socket_ = -1;
#endif
    
    // Initialize AI components
    try {
        orchestral_director_ = std::make_unique<ai::OrchestralAIDirector>();
        model_dispatcher_ = std::make_unique<ai::DynamicModelDispatcher>();
        content_pipeline_ = std::make_unique<ai::RealTimeContentPipeline>();
        quality_engine_ = std::make_unique<ai::AdaptiveQualityEngine>();
        streaming_manager_ = std::make_unique<ai::ProductionStreamingManager>();
        
        // Register all API endpoints
        register_ai_endpoints();
        register_utility_endpoints();
        register_admin_endpoints();
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize REST API Server: " + std::string(e.what()));
    }
}

RestApiServer::~RestApiServer() {
    stop();
#ifdef _WIN32
    WSACleanup();
#endif
}

bool RestApiServer::start() {
    if (running_.load()) {
        return false; // Already running
    }
    
    try {
        // Initialize worker threads
        worker_threads_.reserve(config_.thread_pool_size);
        for (size_t i = 0; i < config_.thread_pool_size; ++i) {
            worker_threads_.emplace_back(&RestApiServer::worker_thread, this);
        }
        
        // Start HTTP server
        server_running_.store(true);
        server_thread_ = std::thread(&RestApiServer::server_loop, this);
        
        running_.store(true);
        return true;
        
    } catch (const std::exception& e) {
        stop();
        return false;
    }
}

void RestApiServer::stop() {
    if (!running_.load()) {
        return;
    }
    
    running_.store(false);
      // Stop HTTP server
    server_running_.store(false);
#ifdef _WIN32
    if (server_socket_ != INVALID_SOCKET) {
        closesocket(server_socket_);
        server_socket_ = INVALID_SOCKET;
    }
#else
    if (server_socket_ != -1) {
        close(server_socket_);
        server_socket_ = -1;
    }
#endif
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    
    // Wait for worker threads to finish
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
}

void RestApiServer::register_ai_endpoints() {
    // Text-to-Image endpoint
    register_endpoint(api::ApiEndpoint(
        api::HttpMethod::POST, "/api/v1/text-to-image",
        [this](const api::HttpRequest& req) { return handle_text_to_image(req); },
        false, "Generate images from text descriptions"
    ));
    
    // Text-to-Speech endpoint
    register_endpoint(ApiEndpoint(
        api::HttpMethod::POST, "/api/v1/text-to-speech",
        [this](const HttpRequest& req) { return handle_text_to_speech(req); },
        false, "Generate speech from text"
    ));
    
    // Text-to-Video endpoint
    register_endpoint(ApiEndpoint(
        api::HttpMethod::POST, "/api/v1/text-to-video",
        [this](const HttpRequest& req) { return handle_text_to_video(req); },
        false, "Generate videos from text descriptions"
    ));
    
    // Multi-modal generation endpoint
    register_endpoint(ApiEndpoint(
        api::HttpMethod::POST, "/api/v1/multimodal-generation",
        [this](const HttpRequest& req) { return handle_multimodal_generation(req); },
        false, "Advanced multi-modal content generation"
    ));
    
    // Orchestral workflow endpoint
    register_endpoint(ApiEndpoint(
        api::HttpMethod::POST, "/api/v1/orchestral-workflow",
        [this](const HttpRequest& req) { return handle_orchestral_workflow(req); },
        true, "Execute complex orchestral AI workflows"
    ));
}

void RestApiServer::register_utility_endpoints() {
    // Health check
    register_endpoint(ApiEndpoint(
        api::HttpMethod::GET, "/api/v1/health",
        [this](const HttpRequest& req) { return handle_health_check(req); },
        false, "Health check endpoint"
    ));
    
    // Metrics
    register_endpoint(ApiEndpoint(
        api::HttpMethod::GET, "/api/v1/metrics",
        [this](const HttpRequest& req) { return handle_metrics(req); },
        false, "Server metrics and statistics"
    ));
      // API documentation
    register_endpoint(ApiEndpoint(
        api::HttpMethod::GET, "/api/v1/docs",
        [this](const HttpRequest& req) { return handle_api_docs(req); },
        false, "API documentation"
    ));
    
    // Model information
    register_endpoint(ApiEndpoint(
        api::HttpMethod::GET, "/api/v1/models",
        [this](const HttpRequest& req) { return handle_model_info(req); },
        false, "Available AI models information"
    ));
}

void RestApiServer::register_admin_endpoints() {    // Admin status
    register_endpoint(ApiEndpoint(
        api::HttpMethod::GET, "/api/v1/admin/status",
        [this](const HttpRequest& req) { return handle_admin_status(req); },
        true, "Administrative status information"
    ));
    
    // Admin configuration
    register_endpoint(ApiEndpoint(
        api::HttpMethod::GET, "/api/v1/admin/config",
        [this](const HttpRequest& req) { return handle_admin_config(req); },
        true, "Server configuration management"
    ));
    
    // Admin authentication
    register_endpoint(ApiEndpoint(
        api::HttpMethod::POST, "/api/v1/admin/auth",
        [this](const HttpRequest& req) { return handle_admin_auth(req); },
        false, "Authentication token management"
    ));
}

void RestApiServer::register_endpoint(const ApiEndpoint& endpoint) {
    std::lock_guard<std::mutex> lock(server_mutex_);
    endpoints_.push_back(endpoint);
}

std::string RestApiServer::create_auth_token(const std::string& user_id, 
                                           const std::vector<std::string>& permissions,
                                           bool is_admin, double expires_in_hours) {
    AuthToken token;
    token.token = generate_token();
    token.user_id = user_id;
    token.expires_at = get_current_time() + (expires_in_hours * 3600.0);
    token.permissions = permissions;
    token.is_admin = is_admin;
    
    std::lock_guard<std::mutex> lock(server_mutex_);
    auth_tokens_[token.token] = token;
    
    return token.token;
}

bool RestApiServer::validate_auth_token(const std::string& token) const {
    // Note: Can't use lock_guard with const method, so we'll use a simple approach
    // In production, this would use a read-write lock or thread-safe container
    auto it = auth_tokens_.find(token);
    return it != auth_tokens_.end() && it->second.is_valid();
}

void RestApiServer::revoke_auth_token(const std::string& token) {
    std::lock_guard<std::mutex> lock(server_mutex_);
    auth_tokens_.erase(token);
}

HttpResponse RestApiServer::handle_request(const HttpRequest& request) {    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Handle CORS preflight
        if (request.method == api::HttpMethod::OPTIONS) {
            return handle_cors_preflight(request);
        }
        
        // Route request to appropriate handler
        HttpResponse response = route_request(request);
        
        // Record metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        bool success = (response.status == api::HttpStatus::OK || response.status == api::HttpStatus::CREATED);
        
        metrics_->record_request(request.path, duration.count(), success);
        
        return response;
        
    } catch (const std::exception& e) {
        HttpResponse error_response;
        error_response.status = api::HttpStatus::INTERNAL_ERROR;
        error_response.body = create_error_response("Internal server error: " + std::string(e.what()));
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        metrics_->record_request(request.path, duration.count(), false);
        
        return error_response;
    }
}

HttpResponse RestApiServer::route_request(const HttpRequest& request) {
    // Find matching endpoint
    for (const auto& endpoint : endpoints_) {        if (endpoint.method == request.method && endpoint.path == request.path) {
            // Check authentication if required
            if (endpoint.requires_auth && !authenticate_request(request)) {
                HttpResponse response;
                response.status = api::HttpStatus::UNAUTHORIZED;
                response.body = create_error_response("Authentication required");
                return response;
            }
            
            // Execute handler
            return endpoint.handler(request);
        }
    }
    
    // No matching endpoint found
    HttpResponse response;
    response.status = HttpStatus::NOT_FOUND;
    response.body = create_error_response("Endpoint not found: " + request.path);
    return response;
}

bool RestApiServer::authenticate_request(const HttpRequest& request) const {
    auto it = request.headers.find(config_.api_key_header);
    if (it == request.headers.end()) {
        return false;
    }
    
    return validate_auth_token(it->second);
}

// AI endpoint handlers
HttpResponse RestApiServer::handle_text_to_image(const HttpRequest& request) {
    HttpResponse response;
    
    try {
        // Parse request body for text prompt
        std::string prompt = "A beautiful landscape"; // Default for demo
        // TODO: Parse JSON request body for actual prompt
        
        // Generate image using AI pipeline
        if (content_pipeline_) {
            // Simulate image generation (placeholder)
            std::string result = create_success_response(
                "{\"image_url\": \"/generated/image_123.png\", \"width\": 512, \"height\": 512}",
                "Image generated successfully"
            );
            response.body = result;
        } else {
            throw std::runtime_error("Content pipeline not available");
        }
        
    } catch (const std::exception& e) {
        response.status = HttpStatus::INTERNAL_ERROR;
        response.body = create_error_response("Text-to-image generation failed: " + std::string(e.what()));
    }
    
    return response;
}

HttpResponse RestApiServer::handle_text_to_speech(const HttpRequest& request) {
    HttpResponse response;
    
    try {
        // Simulate TTS generation
        std::string result = create_success_response(
            "{\"audio_url\": \"/generated/audio_123.wav\", \"duration\": 3.5}",
            "Speech generated successfully"
        );
        response.body = result;
        
    } catch (const std::exception& e) {
        response.status = HttpStatus::INTERNAL_ERROR;
        response.body = create_error_response("Text-to-speech generation failed: " + std::string(e.what()));
    }
    
    return response;
}

HttpResponse RestApiServer::handle_text_to_video(const HttpRequest& request) {
    HttpResponse response;
    
    try {
        // Simulate video generation
        std::string result = create_success_response(
            "{\"video_url\": \"/generated/video_123.mp4\", \"duration\": 10.0, \"fps\": 30}",
            "Video generated successfully"
        );
        response.body = result;
        
    } catch (const std::exception& e) {
        response.status = HttpStatus::INTERNAL_ERROR;
        response.body = create_error_response("Text-to-video generation failed: " + std::string(e.what()));
    }
    
    return response;
}

HttpResponse RestApiServer::handle_multimodal_generation(const HttpRequest& request) {
    HttpResponse response;
    
    try {
        // Simulate multi-modal generation
        std::string result = create_success_response(
            "{\"content\": {\"video\": \"/generated/video_123.mp4\", \"audio\": \"/generated/audio_123.wav\", \"text\": \"Generated content description\"}, \"quality_score\": 0.87}",
            "Multi-modal content generated successfully"
        );
        response.body = result;
        
    } catch (const std::exception& e) {
        response.status = HttpStatus::INTERNAL_ERROR;
        response.body = create_error_response("Multi-modal generation failed: " + std::string(e.what()));
    }
    
    return response;
}

HttpResponse RestApiServer::handle_orchestral_workflow(const HttpRequest& request) {
    HttpResponse response;
    
    try {
        if (orchestral_director_) {
            // Simulate orchestral workflow execution
            std::string result = create_success_response(
                "{\"workflow_id\": \"wf_123\", \"status\": \"completed\", \"execution_time\": 2.34, \"quality_score\": 0.92}",
                "Orchestral workflow executed successfully"
            );
            response.body = result;
        } else {
            throw std::runtime_error("Orchestral director not available");
        }
        
    } catch (const std::exception& e) {
        response.status = HttpStatus::INTERNAL_ERROR;
        response.body = create_error_response("Orchestral workflow failed: " + std::string(e.what()));
    }
    
    return response;
}

// Utility endpoint handlers
HttpResponse RestApiServer::handle_health_check(const HttpRequest& request) {
    HttpResponse response;
    
    std::map<std::string, double> metrics_summary;
    metrics_->get_summary(metrics_summary);
    
    std::ostringstream health_data;
    health_data << "{";
    health_data << "\"status\": \"healthy\",";
    health_data << "\"timestamp\": " << get_current_time() << ",";
    health_data << "\"uptime\": " << (get_current_time() - 0) << ",";
    health_data << "\"version\": \"1.0.0\",";
    health_data << "\"ai_components\": {";
    health_data << "\"orchestral_director\": " << (orchestral_director_ ? "true" : "false") << ",";
    health_data << "\"model_dispatcher\": " << (model_dispatcher_ ? "true" : "false") << ",";
    health_data << "\"content_pipeline\": " << (content_pipeline_ ? "true" : "false") << ",";
    health_data << "\"quality_engine\": " << (quality_engine_ ? "true" : "false") << ",";
    health_data << "\"streaming_manager\": " << (streaming_manager_ ? "true" : "false");
    health_data << "}";
    health_data << "}";
    
    response.body = create_success_response(health_data.str(), "Health check completed");
    return response;
}

HttpResponse RestApiServer::handle_metrics(const HttpRequest& request) {
    HttpResponse response;
    
    std::map<std::string, double> metrics_summary;
    metrics_->get_summary(metrics_summary);
    
    std::ostringstream metrics_json;
    metrics_json << "{";
    bool first = true;
    for (const auto& metric : metrics_summary) {
        if (!first) metrics_json << ",";
        metrics_json << "\"" << metric.first << "\": " << metric.second;
        first = false;
    }
    metrics_json << "}";
    
    response.body = create_success_response(metrics_json.str(), "Metrics retrieved");
    return response;
}

HttpResponse RestApiServer::handle_api_docs(const HttpRequest& request) {
    HttpResponse response;
    response.body = get_api_documentation();
    return response;
}

HttpResponse RestApiServer::handle_model_info(const HttpRequest& request) {
    HttpResponse response;
    
    std::string model_info = create_success_response(
        "{\"models\": [\"text_encoder\", \"image_generator\", \"video_processor\", \"audio_synthesizer\", \"multimodal_fusion\"], \"total_models\": 5}",
        "Model information retrieved"
    );
    response.body = model_info;
    return response;
}

// Admin endpoint handlers
HttpResponse RestApiServer::handle_admin_status(const HttpRequest& request) {
    HttpResponse response;
    response.body = get_health_status();
    return response;
}

HttpResponse RestApiServer::handle_admin_config(const HttpRequest& request) {
    HttpResponse response;
    
    std::ostringstream config_json;
    config_json << "{";
    config_json << "\"host\": \"" << config_.host << "\",";
    config_json << "\"port\": " << config_.port << ",";
    config_json << "\"max_connections\": " << config_.max_connections << ",";
    config_json << "\"thread_pool_size\": " << config_.thread_pool_size << ",";
    config_json << "\"request_timeout\": " << config_.request_timeout << ",";
    config_json << "\"enable_ssl\": " << (config_.enable_ssl ? "true" : "false") << ",";
    config_json << "\"enable_cors\": " << (config_.enable_cors ? "true" : "false");
    config_json << "}";
    
    response.body = create_success_response(config_json.str(), "Configuration retrieved");
    return response;
}

HttpResponse RestApiServer::handle_admin_auth(const HttpRequest& request) {
    HttpResponse response;
    
    if (request.method == HttpMethod::POST) {
        // Create new token
        std::string token = create_auth_token("admin_user", {"read", "write"}, true);
        std::string result = create_success_response(
            "{\"token\": \"" + token + "\", \"expires_in\": 86400}",
            "Authentication token created"
        );
        response.body = result;
    } else {
        response.status = HttpStatus::BAD_REQUEST;
        response.body = create_error_response("Invalid method for authentication endpoint");
    }
    
    return response;
}

HttpResponse RestApiServer::handle_cors_preflight(const HttpRequest& request) {
    HttpResponse response;
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS";
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key";
    response.headers["Access-Control-Max-Age"] = "86400";
    response.body = "";
    return response;
}

// HTTP Server Implementation
void RestApiServer::server_loop() {
    // Create socket
    server_socket_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
#ifdef _WIN32
    if (server_socket_ == INVALID_SOCKET) {
        std::cerr << "Socket creation failed: " << WSAGetLastError() << std::endl;
        return;
    }
#else
    if (server_socket_ < 0) {
        std::cerr << "Socket creation failed" << std::endl;
        return;
    }
#endif
    
    // Bind socket
    sockaddr_in server_addr = {};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(static_cast<unsigned short>(config_.port));
    
#ifdef _WIN32
    if (bind(server_socket_, (sockaddr*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR) {
        std::cerr << "Bind failed: " << WSAGetLastError() << std::endl;
        closesocket(server_socket_);
        return;
    }
    
    // Listen for connections
    if (listen(server_socket_, 10) == SOCKET_ERROR) {
        std::cerr << "Listen failed: " << WSAGetLastError() << std::endl;
        closesocket(server_socket_);
        return;
    }
#else
    if (bind(server_socket_, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Bind failed" << std::endl;
        close(server_socket_);
        return;
    }
    
    // Listen for connections
    if (listen(server_socket_, 10) < 0) {
        std::cerr << "Listen failed" << std::endl;
        close(server_socket_);
        return;
    }
#endif
    
    std::cout << "REST API Server listening on port " << config_.port << std::endl;
    
    while (server_running_.load()) {
        // Accept client connection
#ifdef _WIN32
        SOCKET client_socket = accept(server_socket_, nullptr, nullptr);
        if (client_socket == INVALID_SOCKET) {
            if (server_running_.load()) {
                std::cerr << "Accept failed: " << WSAGetLastError() << std::endl;
            }
            continue;
        }
#else
        int client_socket = accept(server_socket_, nullptr, nullptr);
        if (client_socket < 0) {
            if (server_running_.load()) {
                std::cerr << "Accept failed" << std::endl;
            }
            continue;
        }
#endif
        
        // Handle client in separate thread
        std::thread client_thread(&RestApiServer::handle_client_connection, this, client_socket);
        client_thread.detach();
    }
    
#ifdef _WIN32
    closesocket(server_socket_);
#else
    close(server_socket_);
#endif
}

#ifdef _WIN32
void RestApiServer::handle_client_connection(SOCKET client_socket) {
#else
void RestApiServer::handle_client_connection(int client_socket) {
#endif
    char buffer[8192];
#ifdef _WIN32
    int bytes_received = recv(client_socket, buffer, sizeof(buffer) - 1, 0);
#else
    ssize_t bytes_received = recv(client_socket, buffer, sizeof(buffer) - 1, 0);
#endif
    
    if (bytes_received > 0) {
        buffer[bytes_received] = '\0';
        std::string raw_request(buffer);
        
        // Parse HTTP request
        HttpRequest request = parse_http_request(raw_request);
        
        // Find matching endpoint
        HttpResponse response;
        bool endpoint_found = false;
        
        for (const auto& endpoint : endpoints_) {
            if (endpoint.method == request.method && endpoint.path == request.path) {
                // Check authentication if required
                if (endpoint.requires_auth && !validate_auth_token(request.headers.find("Authorization") != request.headers.end() ? request.headers.at("Authorization") : "")) {
                    response.status = HttpStatus::UNAUTHORIZED;
                    response.body = "{\"error\":\"Authentication required\"}";
                } else {
                    // Call endpoint handler
                    auto start_time = std::chrono::high_resolution_clock::now();
                    try {
                        response = endpoint.handler(request);
                        auto end_time = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);                        metrics_->record_request(endpoint.path, static_cast<double>(duration.count()) / 1000.0, true);
                    } catch (const std::exception& e) {
                        response.status = api::HttpStatus::INTERNAL_ERROR;
                        response.body = "{\"error\":\"" + utils::escape_json(e.what()) + "\"}";
                        auto end_time = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                        metrics_->record_request(endpoint.path, static_cast<double>(duration.count()) / 1000.0, false);
                    }
                }
                endpoint_found = true;
                break;
            }
        }
        
        if (!endpoint_found) {
            response.status = HttpStatus::NOT_FOUND;
            response.body = "{\"error\":\"Endpoint not found\"}";
        }
        
        // Send response
        std::string http_response = format_http_response(response);
#ifdef _WIN32
        send(client_socket, http_response.c_str(), static_cast<int>(http_response.length()), 0);
    }
    
    closesocket(client_socket);
#else
        send(client_socket, http_response.c_str(), http_response.length(), 0);
    }
    
    close(client_socket);
#endif
}

HttpRequest RestApiServer::parse_http_request(const std::string& raw_request) {
    HttpRequest request;
    std::istringstream stream(raw_request);
    std::string line;
    
    // Parse request line
    if (std::getline(stream, line)) {
        std::istringstream request_line(line);
        std::string method_str, path, version;
        request_line >> method_str >> path >> version;
        
        // Parse method
        if (method_str == "GET") request.method = HttpMethod::GET;
        else if (method_str == "POST") request.method = HttpMethod::POST;
        else if (method_str == "PUT") request.method = HttpMethod::PUT;
        else if (method_str == "DELETE") request.method = HttpMethod::DELETE;
        else if (method_str == "OPTIONS") request.method = HttpMethod::OPTIONS;
        
        // Parse path and query parameters
        size_t query_pos = path.find('?');
        if (query_pos != std::string::npos) {
            request.path = path.substr(0, query_pos);
            std::string query_string = path.substr(query_pos + 1);
            // Parse query parameters (simplified)
            // TODO: Full URL decoding and parameter parsing
        } else {
            request.path = path;
        }
    }
    
    // Parse headers
    while (std::getline(stream, line) && line != "\r" && !line.empty()) {
        size_t colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
            std::string name = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 1);
            // Trim whitespace
            size_t start = value.find_first_not_of(" \t");
            size_t end = value.find_last_not_of(" \t\r\n");
            if (start != std::string::npos && end != std::string::npos) {
                value = value.substr(start, end - start + 1);
            }
            request.headers[name] = value;
        }
    }
    
    // Parse body
    std::string body;
    std::string body_line;
    while (std::getline(stream, body_line)) {
        body += body_line + "\n";
    }
    if (!body.empty() && body.back() == '\n') {
        body.pop_back(); // Remove trailing newline
    }
    request.body = body;
    
    return request;
}

std::string RestApiServer::format_http_response(const HttpResponse& response) {
    std::ostringstream stream;
    
    // Status line
    stream << "HTTP/1.1 " << static_cast<int>(response.status);
    switch (response.status) {
        case HttpStatus::OK: stream << " OK"; break;
        case HttpStatus::CREATED: stream << " Created"; break;
        case HttpStatus::BAD_REQUEST: stream << " Bad Request"; break;
        case HttpStatus::UNAUTHORIZED: stream << " Unauthorized"; break;
        case HttpStatus::NOT_FOUND: stream << " Not Found"; break;
        case HttpStatus::INTERNAL_ERROR: stream << " Internal Server Error"; break;
    }
    stream << "\r\n";
    
    // Headers
    for (const auto& header : response.headers) {
        stream << header.first << ": " << header.second << "\r\n";
    }
    stream << "Content-Length: " << response.body.length() << "\r\n";
    stream << "Connection: close\r\n";
    stream << "\r\n";
    
    // Body
    stream << response.body;
    
    return stream.str();
}

std::string RestApiServer::get_content_type(const std::string& path) {
    if (path.find(".json") != std::string::npos) return "application/json";
    if (path.find(".html") != std::string::npos) return "text/html";
    if (path.find(".css") != std::string::npos) return "text/css";
    if (path.find(".js") != std::string::npos) return "application/javascript";
    return "application/json"; // Default for API
}

// Worker thread implementation
void RestApiServer::worker_thread() {
    while (running_.load()) {
        try {
            // Simulate HTTP request processing
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } catch (const std::exception& e) {
            // Log error and continue
        }
    }
}

// Utility functions
std::string RestApiServer::generate_token() const {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    
    std::string token;
    for (int i = 0; i < 32; ++i) {
        token += "0123456789abcdef"[dis(gen)];
    }
    return token;
}

double RestApiServer::get_current_time() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    return static_cast<double>(time_t);
}

std::string RestApiServer::create_error_response(const std::string& message, const std::string& error_code) const {
    std::ostringstream json;
    json << "{";
    json << "\"success\": false,";
    json << "\"error\": {";
    json << "\"message\": \"" << utils::escape_json(message) << "\"";
    if (!error_code.empty()) {
        json << ",\"code\": \"" << utils::escape_json(error_code) << "\"";
    }
    json << "},";
    json << "\"timestamp\": " << get_current_time();
    json << "}";
    return json.str();
}

std::string RestApiServer::create_success_response(const std::string& data, const std::string& message) const {
    std::ostringstream json;
    json << "{";
    json << "\"success\": true,";
    json << "\"message\": \"" << utils::escape_json(message) << "\",";
    json << "\"data\": " << data << ",";
    json << "\"timestamp\": " << get_current_time();
    json << "}";
    return json.str();
}

std::string RestApiServer::get_health_status() const {
    std::map<std::string, double> metrics_summary;
    metrics_->get_summary(metrics_summary);
    
    std::ostringstream health;
    health << "{";
    health << "\"server_status\": \"" << (running_.load() ? "running" : "stopped") << "\",";
    health << "\"thread_count\": " << worker_threads_.size() << ",";
    health << "\"endpoint_count\": " << endpoints_.size() << ",";
    health << "\"auth_tokens\": " << auth_tokens_.size() << ",";
    health << "\"metrics\": {";
    
    bool first = true;
    for (const auto& metric : metrics_summary) {
        if (!first) health << ",";
        health << "\"" << metric.first << "\": " << metric.second;
        first = false;
    }
    
    health << "}";
    health << "}";
    
    return create_success_response(health.str(), "Admin status retrieved");
}

std::string RestApiServer::get_api_documentation() const {
    std::ostringstream docs;
    docs << "{";
    docs << "\"title\": \"CLModel REST API\",";
    docs << "\"version\": \"1.0.0\",";
    docs << "\"description\": \"Enterprise-grade AI API for CLModel framework\",";
    docs << "\"endpoints\": [";
    
    bool first = true;
    for (const auto& endpoint : endpoints_) {
        if (!first) docs << ",";
        docs << "{";
        docs << "\"method\": \"" << utils::http_method_to_string(endpoint.method) << "\",";
        docs << "\"path\": \"" << endpoint.path << "\",";
        docs << "\"description\": \"" << utils::escape_json(endpoint.description) << "\",";
        docs << "\"requires_auth\": " << (endpoint.requires_auth ? "true" : "false");
        docs << "}";
        first = false;
    }
    
    docs << "]";
    docs << "}";
    
    return create_success_response(docs.str(), "API documentation retrieved");
}

// Utility namespace implementation
namespace utils {

std::string http_method_to_string(HttpMethod method) {
    switch (method) {
        case HttpMethod::GET: return "GET";
        case HttpMethod::POST: return "POST";
        case HttpMethod::PUT: return "PUT";
        case HttpMethod::DELETE: return "DELETE";
        case HttpMethod::OPTIONS: return "OPTIONS";
        default: return "UNKNOWN";
    }
}

HttpMethod string_to_http_method(const std::string& method) {
    if (method == "GET") return HttpMethod::GET;
    if (method == "POST") return HttpMethod::POST;
    if (method == "PUT") return HttpMethod::PUT;
    if (method == "DELETE") return HttpMethod::DELETE;
    if (method == "OPTIONS") return HttpMethod::OPTIONS;
    return HttpMethod::GET; // Default
}

std::string http_status_to_string(HttpStatus status) {
    switch (status) {
        case HttpStatus::OK: return "200 OK";
        case HttpStatus::CREATED: return "201 Created";
        case HttpStatus::BAD_REQUEST: return "400 Bad Request";
        case HttpStatus::UNAUTHORIZED: return "401 Unauthorized";
        case HttpStatus::NOT_FOUND: return "404 Not Found";
        case HttpStatus::INTERNAL_ERROR: return "500 Internal Server Error";
        default: return "500 Internal Server Error";
    }
}

std::string escape_json(const std::string& str) {
    std::string escaped;
    for (char c : str) {
        switch (c) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += c; break;
        }
    }
    return escaped;
}

bool is_valid_json(const std::string& json) {
    // Simple JSON validation (basic implementation)
    if (json.empty()) return false;
    
    std::stack<char> brackets;
    bool in_string = false;
    bool escaped = false;
    
    for (char c : json) {
        if (escaped) {
            escaped = false;
            continue;
        }
        
        if (c == '\\') {
            escaped = true;
            continue;
        }
        
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        
        if (in_string) continue;
        
        if (c == '{' || c == '[') {
            brackets.push(c);
        } else if (c == '}') {
            if (brackets.empty() || brackets.top() != '{') return false;
            brackets.pop();
        } else if (c == ']') {
            if (brackets.empty() || brackets.top() != '[') return false;
            brackets.pop();
        }
    }
    
    return brackets.empty() && !in_string;
}

} // namespace utils

} // namespace clmodel
