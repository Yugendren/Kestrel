#ifndef KYTY_COMMON_SYSTEM_INFO_H_
#define KYTY_COMMON_SYSTEM_INFO_H_

#include <cstdint>
#include <string>

namespace Common {

struct SystemInfo {
	std::string ProcessorName;
	uint32_t    PhysicalCores  = 0;
	uint32_t    LogicalThreads = 0; // SMT siblings counted separately
};

[[nodiscard]] SystemInfo GetSystemInfo();

} // namespace Common

#endif /* KYTY_COMMON_SYSTEM_INFO_H_ */
