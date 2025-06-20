#include <iostream>
#include <thread>
#include <chrono>
#include "api/rest_api_server.hpp"

int main() {
    std::cout << "=== CLModel Week 17: REST API Server Demo ===" << std::endl;
    
    try {
        // Configure REST API server
        clmodel::api::RestApiServer::Config config;
        config.host = "localhost";
        config.port = 8080;
        config.thread_pool_size = 4;
        config.max_connections = 50;
        
        std::cout << "Creating REST API server..." << std::endl;
        clmodel::api::RestApiServer server(config);
        
        std::cout << "Starting REST API server on " << config.host << ":" << config.port << std::endl;
        if (!server.start()) {
            std::cerr << "Failed to start REST API server" << std::endl;
            return 1;
        }
        
        std::cout << "REST API server started successfully!" << std::endl;
        std::cout << "Available endpoints:" << std::endl;
        std::cout << "  GET  /api/v1/health          - Health check" << std::endl;
        std::cout << "  GET  /api/v1/metrics         - Server metrics" << std::endl;
        std::cout << "  GET  /api/v1/docs            - API documentation" << std::endl;
        std::cout << "  GET  /api/v1/models          - Available AI models" << std::endl;
        std::cout << "  POST /api/v1/text-to-image   - Generate images from text" << std::endl;
        std::cout << "  POST /api/v1/text-to-speech  - Generate speech from text" << std::endl;
        std::cout << "  POST /api/v1/text-to-video   - Generate videos from text" << std::endl;
        std::cout << "  POST /api/v1/multimodal-generation - Multi-modal content generation" << std::endl;
        std::cout << std::endl;
        
        std::cout << "You can test the API using curl or a web browser:" << std::endl;
        std::cout << "  curl http://localhost:8080/api/v1/health" << std::endl;
        std::cout << "  curl http://localhost:8080/api/v1/metrics" << std::endl;
        std::cout << std::endl;
        
        std::cout << "Press Enter to stop the server..." << std::endl;
        std::cin.get();
        
        std::cout << "Stopping REST API server..." << std::endl;
        server.stop();
        std::cout << "REST API server stopped." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "=== Week 17 REST API Server Demo Complete ===" << std::endl;
    return 0;
}
