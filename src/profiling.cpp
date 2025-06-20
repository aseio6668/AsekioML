#include "profiling.hpp"

namespace clmodel {
namespace profiling {

// Static instance
std::unique_ptr<Profiler> Profiler::instance_ = nullptr;

} // namespace profiling
} // namespace clmodel
