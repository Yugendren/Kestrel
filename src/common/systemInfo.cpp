#include "common/systemInfo.h"

#include "common/assert.h"
#include "cpuinfo.h"

namespace Common {

SystemInfo GetSystemInfo() {
	static const bool initialized = cpuinfo_initialize();
	EXIT_IF(!initialized);

	const auto* package = cpuinfo_get_package(0);
	EXIT_IF(package == nullptr);

	SystemInfo info;
	info.ProcessorName  = package->name;
	info.PhysicalCores  = cpuinfo_get_cores_count();
	info.LogicalThreads = cpuinfo_get_processors_count();
	return info;
}

} // namespace Common
