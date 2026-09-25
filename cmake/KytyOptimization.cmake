# Release-performance switches: target ISA, floating-point contraction, ThinLTO, instrumented PGO.
#
# Included from the top-level CMakeLists.txt after config_compiler_and_linker() and before the
# first add_subdirectory(), so the directory-scoped options below reach every target, including
# the third-party libraries that sit on the command processor's hot path (xxhash, fmt, spdlog).
#
# Options (all cache variables, override with -D on the configure line):
#   KYTY_MARCH        ISA level, e.g. x86-64-v3 (default on x86-64 hosts other than macOS), or
#                     empty for the compiler's baseline.
#   KYTY_LTO          ThinLTO for every non-Debug configuration (Clang only).
#   KYTY_PGO          OFF | GENERATE | USE: instrumented profile-guided optimisation (Clang only).
#   KYTY_PGO_PROFILE  merged .profdata consumed by KYTY_PGO=USE.
# Tracy is switched separately by KYTY_TRACY in the top-level CMakeLists.txt.

include_guard(GLOBAL)

# Adds a compiler option in the spelling the active driver understands: clang-cl only forwards
# GCC-style options it does not know through /clang:.
function(kyty_add_driver_compile_option opt)
	if(KYTY_CLANG_CL)
		add_compile_options("/clang:${opt}")
	else()
		add_compile_options("${opt}")
	endif()
endfunction()

# --- Target ISA -------------------------------------------------------------------------------
# PS5 guest code is Zen 2 machine code that runs natively on the host, so every machine able to
# run a title already has AVX2/BMI2/FMA and x86-64-v3 costs no users. It buys inline roundsd for
# floor/ceil, tzcnt/popcnt in the bit arrays, 32-byte inline copies and XXH3's AVX2 accumulator.
# macOS stays at baseline: Rosetta 2 translates AVX2 only from macOS 15 on.
if(APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
	set(KYTY_MARCH_DEFAULT "")
else()
	set(KYTY_MARCH_DEFAULT "x86-64-v3")
endif()
set(KYTY_MARCH "${KYTY_MARCH_DEFAULT}" CACHE STRING
	"Host ISA level passed as -march (empty = compiler baseline)")

if(CLANG OR GCC)
	if(NOT KYTY_MARCH STREQUAL "")
		kyty_add_driver_compile_option("-march=${KYTY_MARCH}")
	endif()
	# Never fuse a*b+c into an FMA. With FMA available Clang's default (-ffp-contract=on, and
	# /fp:precise under clang-cl) changes float results bit for bit, e.g. the viewport/scissor math
	# that feeds floor/lround when a draw is prepared. Off keeps every build, with or without
	# KYTY_MARCH, numerically identical to the baseline x86-64 build.
	kyty_add_driver_compile_option("-ffp-contract=off")
endif()

# --- ThinLTO ----------------------------------------------------------------------------------
# The command processor's per-draw path crosses ~40 call sites in a dozen translation units plus
# libcommon; ThinLTO lets the small accessors among them inline. Debug builds are left alone so the
# edit-compile loop keeps its link time.
option(KYTY_LTO "ThinLTO for non-Debug builds (Clang)" ON)

if(KYTY_LTO AND CLANG AND NOT APPLE)
	include(CheckIPOSupported)
	check_ipo_supported(RESULT KYTY_LTO_SUPPORTED OUTPUT KYTY_LTO_ERROR LANGUAGES C CXX)
	if(KYTY_LTO_SUPPORTED)
		# CMake's Clang IPO support means -flto=thin plus llvm-ar/llvm-ranlib for the archives.
		# Third-party projects with an old cmake_minimum_required (SDL2) would otherwise ignore
		# the property under policy CMP0069 OLD.
		set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
		set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
		set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
		set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ON)
		if(KYTY_LINKER STREQUAL LLD)
			# Every executable (tests too) must link with lld: the system ld cannot read bitcode.
			# The cache makes a relink after a one-file edit re-run the backend for that module only.
			add_link_options("-fuse-ld=lld" "LINKER:--thinlto-cache-dir=${CMAKE_BINARY_DIR}/thinlto-cache")
		elseif(KYTY_LINKER STREQUAL LLD_LINK)
			add_link_options("/lldltocache:${CMAKE_BINARY_DIR}/thinlto-cache")
		endif()
	else()
		message(WARNING "KYTY_LTO requested but not supported by this toolchain: ${KYTY_LTO_ERROR}")
	endif()
endif()

# --- Instrumented PGO -------------------------------------------------------------------------
# Zen 2 has no LBR, so sampling PGO (AutoFDO/CSSPGO/Propeller) is unavailable; IR instrumentation
# needs no PMU. GENERATE builds a counting binary that writes kyty-<pid>.profraw into the working
# directory (see the periodic dump in main.cpp: benches kill the emulator, so the at-exit write
# never happens). Merge the files with llvm-profdata and rebuild with KYTY_PGO=USE.
# Profiles are toolchain- and ABI-specific: the Windows (clang-cl) build needs its own training run.
set(KYTY_PGO "OFF" CACHE STRING "Profile-guided optimisation: OFF, GENERATE or USE (Clang)")
set_property(CACHE KYTY_PGO PROPERTY STRINGS OFF GENERATE USE)
set(KYTY_PGO_PROFILE "" CACHE FILEPATH "Merged .profdata file used by KYTY_PGO=USE")

if(NOT KYTY_PGO STREQUAL "OFF")
	if(NOT CLANG)
		message(FATAL_ERROR "KYTY_PGO requires Clang")
	endif()
	if(KYTY_PGO STREQUAL "GENERATE")
		kyty_add_driver_compile_option("-fprofile-generate")
		add_compile_definitions(KYTY_PGO_GENERATE=1)
		if(KYTY_CLANG_CL)
			# clang-cl records the profile runtime as a /DEFAULTLIB in each object; lld-link only
			# needs to be told where compiler-rt lives.
			execute_process(COMMAND "${CMAKE_CXX_COMPILER}" /clang:-print-resource-dir
				OUTPUT_VARIABLE KYTY_CLANG_RESOURCE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
			# LLVM <= 18 installs it as lib/windows/clang_rt.profile-x86_64.lib, newer releases
			# use the per-target directory; both names are what clang-cl records.
			foreach(dir "lib/windows" "lib/x86_64-pc-windows-msvc")
				if(EXISTS "${KYTY_CLANG_RESOURCE_DIR}/${dir}")
					add_link_options("/LIBPATH:${KYTY_CLANG_RESOURCE_DIR}/${dir}")
				endif()
			endforeach()
		else()
			add_link_options("-fprofile-generate")
		endif()
	elseif(KYTY_PGO STREQUAL "USE")
		if(NOT EXISTS "${KYTY_PGO_PROFILE}")
			message(FATAL_ERROR "KYTY_PGO=USE needs KYTY_PGO_PROFILE pointing at a merged .profdata")
		endif()
		kyty_add_driver_compile_option("-fprofile-use=${KYTY_PGO_PROFILE}")
		# Code that the training run never reached (other titles, error paths) is expected; do not
		# warn once per function about it.
		add_compile_options(-Wno-profile-instr-unprofiled -Wno-profile-instr-out-of-date)
		# Ninja does not see the profile as an input; a changed profile needs a clean rebuild.
		message(STATUS "KYTY_PGO=USE with ${KYTY_PGO_PROFILE} (clean-rebuild after replacing it)")
	else()
		message(FATAL_ERROR "KYTY_PGO must be OFF, GENERATE or USE")
	endif()
endif()

message(STATUS "Kyty optimisation: march='${KYTY_MARCH}' lto=${KYTY_LTO} pgo=${KYTY_PGO}")
