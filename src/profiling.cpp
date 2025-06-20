#include "profiling.hpp"

namespace asekioml {
namespace profiling {

// Static instance
std::unique_ptr<Profiler> Profiler::instance_ = nullptr;

} // namespace profiling
} // namespace asekioml
