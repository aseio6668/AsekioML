#pragma once

#include <string>
#include <memory>
#include <functional>
#include <map>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <future>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#endif

// Forward declarations for CLModel AI components
namespace clmodel {
    class Tensor;
    class Matrix;
    
    namespace ai {
        class OrchestralAIDirector;
        class DynamicModelDispatcher;
        class RealTimeContentPipeline;
        class AdaptiveQualityEngine;
        class ProductionStreamingManager;
    }
    
    namespace api {
        
        // HTTP Methods
        enum class HttpMethod {
            GET,
            POST,
            PUT,
            DELETE,
            OPTIONS
        };

        // HTTP Response Status Codes
        enum class HttpStatus {
            OK = 200,
            CREATED = 201,
            BAD_REQUEST = 400,
            UNAUTHORIZED = 401,
            NOT_FOUND = 404,
            INTERNAL_ERROR = 500
        };

        // Request structure
        struct HttpRequest {
            HttpMethod method;
            std::string path;
            std::map<std::string, std::string> headers;
            std::map<std::string, std::string> query_params;
            std::string body;
            std::string remote_address;
            double timestamp;
        };

        // Response structure
        struct HttpResponse {
            HttpStatus status;
            std::map<std::string, std::string> headers;
            std::string body;
            size_t content_length;
            
            HttpResponse() : status(HttpStatus::OK), content_length(0) {
                headers["Content-Type"] = "application/json";
                headers["Access-Control-Allow-Origin"] = "*";
            }
        };

        // API endpoint handler function type
        using ApiHandler = std::function<HttpResponse(const HttpRequest&)>;

        // API endpoint registration structure
        struct ApiEndpoint {
            HttpMethod method;
            std::string path;
            ApiHandler handler;
            bool requires_auth;
            std::string description;
            
            ApiEndpoint(HttpMethod m, const std::string& p, ApiHandler h, 
                        bool auth = false, const std::string& desc = "")
                : method(m), path(p), handler(h), requires_auth(auth), description(desc) {}
        };

    } // namespace api

    // Authentication token structure
    struct AuthToken {
        std::string token;
        std::string user_id;
        double expires_at;
        std::vector<std::string> permissions;
        bool is_admin;
        
        bool is_valid() const {
            return !token.empty() && expires_at > get_current_time();
        }
        
    private:
        double get_current_time() const;
    };

    // Request/Response logging and metrics
    struct RequestMetrics {
        std::atomic<size_t> total_requests{0};
        std::atomic<size_t> successful_requests{0};
        std::atomic<size_t> failed_requests{0};
        std::atomic<double> average_response_time{0.0};
        std::atomic<size_t> active_connections{0};
        std::map<std::string, std::atomic<size_t>> endpoint_counts;
        std::mutex metrics_mutex;
        
        void record_request(const std::string& endpoint, double response_time, bool success);
        void get_summary(std::map<std::string, double>& summary) const;
    };

    // Main REST API Server class
    class RestApiServer {
    public:
        struct Config {
            std::string host = "0.0.0.0";
            int port = 8080;
            size_t max_connections = 100;
            size_t thread_pool_size = 8;
            double request_timeout = 30.0;
            bool enable_ssl = false;
            std::string ssl_cert_path;
            std::string ssl_key_path;
            bool enable_cors = true;
            bool enable_compression = true;
            std::string api_key_header = "X-API-Key";
            size_t max_request_size = 10 * 1024 * 1024; // 10MB
        };

    private:
        Config config_;
        std::unique_ptr<ai::OrchestralAIDirector> orchestral_director_;
        std::unique_ptr<ai::DynamicModelDispatcher> model_dispatcher_;
        std::unique_ptr<ai::RealTimeContentPipeline> content_pipeline_;
        std::unique_ptr<ai::AdaptiveQualityEngine> quality_engine_;
        std::unique_ptr<ai::ProductionStreamingManager> streaming_manager_;
        
        std::vector<api::ApiEndpoint> endpoints_;
        std::map<std::string, AuthToken> auth_tokens_;
        std::unique_ptr<RequestMetrics> metrics_;
        
        std::atomic<bool> running_{false};
        std::vector<std::thread> worker_threads_;
        std::mutex server_mutex_;
        
        // HTTP server socket management
    #ifdef _WIN32
        SOCKET server_socket_;
    #else
        int server_socket_;
    #endif
        std::atomic<bool> server_running_;
        std::thread server_thread_;
        
    public:
        RestApiServer(const Config& config = Config{});
        ~RestApiServer();

        // Server lifecycle
        bool start();
        void stop();
        bool is_running() const { return running_.load(); }

        // Endpoint registration
        void register_endpoint(const api::ApiEndpoint& endpoint);
        void register_ai_endpoints();
        void register_utility_endpoints();
        void register_admin_endpoints();

        // Authentication
        std::string create_auth_token(const std::string& user_id, 
                                     const std::vector<std::string>& permissions = {},
                                     bool is_admin = false,
                                     double expires_in_hours = 24.0);
        bool validate_auth_token(const std::string& token) const;
        void revoke_auth_token(const std::string& token);

        // Request handling
        api::HttpResponse handle_request(const api::HttpRequest& request);
        api::HttpResponse handle_cors_preflight(const api::HttpRequest& request);
        
        // Metrics and monitoring
        RequestMetrics& get_metrics() { return *metrics_; }
        std::string get_health_status() const;
        std::string get_api_documentation() const;

    private:
        // Core HTTP server implementation
        void server_loop();
    #ifdef _WIN32
        void handle_client_connection(SOCKET client_socket);
    #else
        void handle_client_connection(int client_socket);
    #endif
        api::HttpRequest parse_http_request(const std::string& raw_request);
        std::string format_http_response(const api::HttpResponse& response);
        std::string get_content_type(const std::string& path);
        api::HttpResponse route_request(const api::HttpRequest& request);
        bool authenticate_request(const api::HttpRequest& request) const;
        
        // AI endpoint handlers
        api::HttpResponse handle_text_to_image(const api::HttpRequest& request);
        api::HttpResponse handle_text_to_speech(const api::HttpRequest& request);
        api::HttpResponse handle_text_to_video(const api::HttpRequest& request);
        api::HttpResponse handle_multimodal_generation(const api::HttpRequest& request);
        api::HttpResponse handle_orchestral_workflow(const api::HttpRequest& request);
        
        // Utility endpoint handlers
        api::HttpResponse handle_health_check(const api::HttpRequest& request);
        api::HttpResponse handle_metrics(const api::HttpRequest& request);
        api::HttpResponse handle_api_docs(const api::HttpRequest& request);
        api::HttpResponse handle_model_info(const api::HttpRequest& request);
        
        // Admin endpoint handlers
        api::HttpResponse handle_admin_status(const api::HttpRequest& request);
        api::HttpResponse handle_admin_config(const api::HttpRequest& request);
        api::HttpResponse handle_admin_auth(const api::HttpRequest& request);
        
        // JSON serialization helpers
        std::string tensor_to_json(const Tensor& tensor) const;
        std::string matrix_to_json(const Matrix& matrix) const;
        Tensor json_to_tensor(const std::string& json) const;
        Matrix json_to_matrix(const std::string& json) const;
        
        // Utility functions
        std::string generate_token() const;
        double get_current_time() const;
        std::string encode_base64(const std::vector<uint8_t>& data) const;
        std::vector<uint8_t> decode_base64(const std::string& encoded) const;
        std::string create_error_response(const std::string& message, 
                                        const std::string& error_code = "") const;
        std::string create_success_response(const std::string& data, 
                                          const std::string& message = "Success") const;
    };

    // Utility functions for API development
    namespace utils {
        std::string http_method_to_string(api::HttpMethod method);
        api::HttpMethod string_to_http_method(const std::string& method);
        std::string http_status_to_string(api::HttpStatus status);
        std::string url_encode(const std::string& str);
        std::string url_decode(const std::string& str);
        std::map<std::string, std::string> parse_query_params(const std::string& query);
        std::string escape_json(const std::string& str);
        bool is_valid_json(const std::string& json);
    }

} // namespace clmodel
