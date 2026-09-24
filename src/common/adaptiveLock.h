#ifndef KYTY_COMMON_ADAPTIVELOCK_H_
#define KYTY_COMMON_ADAPTIVELOCK_H_

#include "common/common.h"

#include <atomic>
#include <cstdint>

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#include <cerrno>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace Common {

// Spin-then-block mutex (BasicLockable). A pure spin lock degrades badly once more threads
// contend than there are cores: a preempted holder leaves every waiter burning its whole
// timeslice. This lock spins briefly for the common short critical section, then sleeps in
// the kernel (futex on Linux, WaitOnAddress via std::atomic::wait elsewhere).
//
// Three-state protocol: 0 = unlocked, 1 = locked, 2 = locked and possibly contended. Unlock
// only makes a wake syscall when the state was 2, so the uncontended path is two atomic RMWs.
//
// Usable from the synchronous SIGSEGV handler: no allocation, no thread-local state, and the
// Linux path talks to the futex syscall directly instead of the C++ runtime's waiter tables.
class AdaptiveLock final {
public:
	AdaptiveLock() = default;
	~AdaptiveLock() = default;
	KYTY_CLASS_NO_COPY(AdaptiveLock);

	[[nodiscard]] bool try_lock() noexcept {
		uint32_t expected = UNLOCKED;
		return m_state.compare_exchange_strong(expected, LOCKED, std::memory_order_acquire,
		                                       std::memory_order_relaxed);
	}

	void lock() noexcept {
		if (try_lock()) {
			return;
		}
		LockSlow();
	}

	void unlock() noexcept {
		if (m_state.exchange(UNLOCKED, std::memory_order_release) == CONTENDED) {
			Wake();
		}
	}

private:
	static constexpr uint32_t UNLOCKED  = 0;
	static constexpr uint32_t LOCKED    = 1;
	static constexpr uint32_t CONTENDED = 2;

	// Enough to cover a short critical section held by a running thread without paying for a
	// kernel round trip; small enough that a preempted holder costs microseconds, not a slice.
	static constexpr int SPIN_COUNT = 128;

	static void Pause() noexcept {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
		_mm_pause();
#else
		std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
	}

	void LockSlow() noexcept {
		for (int i = 0; i < SPIN_COUNT; i++) {
			Pause();
			// Test before the RMW so spinners do not keep stealing the cache line from the holder.
			if (m_state.load(std::memory_order_relaxed) == UNLOCKED && try_lock()) {
				return;
			}
		}
		// Once we mark the lock contended we must keep acquiring it as CONTENDED: we cannot know
		// whether other sleepers remain, so the eventual unlock has to issue a wake.
		while (m_state.exchange(CONTENDED, std::memory_order_acquire) != UNLOCKED) {
			Wait();
		}
	}

	void Wait() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
		// Returns immediately (EAGAIN) if the state already changed; spurious wakeups and EINTR
		// are handled by the caller's retry loop. errno is preserved because this can run inside
		// the fault handler, interrupting code that is about to read errno.
		const int saved_errno = errno;
		::syscall(SYS_futex, reinterpret_cast<uint32_t*>(&m_state), FUTEX_WAIT_PRIVATE, CONTENDED,
		          nullptr, nullptr, 0);
		errno = saved_errno;
#else
		m_state.wait(CONTENDED, std::memory_order_relaxed);
#endif
	}

	void Wake() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
		::syscall(SYS_futex, reinterpret_cast<uint32_t*>(&m_state), FUTEX_WAKE_PRIVATE, 1, nullptr,
		          nullptr, 0);
#else
		m_state.notify_one();
#endif
	}

	std::atomic<uint32_t> m_state {UNLOCKED};
};

static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t));
static_assert(std::atomic<uint32_t>::is_always_lock_free);

} // namespace Common

#endif /* KYTY_COMMON_ADAPTIVELOCK_H_ */
