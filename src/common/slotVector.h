#ifndef EMULATOR_SRC_COMMON_SLOTVECTOR_H_
#define EMULATOR_SRC_COMMON_SLOTVECTOR_H_

#include "common/assert.h"
#include "common/abi.h"
#include "common/common.h"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace Common {

struct SlotId {
	static constexpr uint32_t INVALID_INDEX = std::numeric_limits<uint32_t>::max();

	constexpr SlotId() noexcept = default;
	constexpr SlotId(uint32_t value) noexcept: index(value), generation(1) {}
	constexpr SlotId(uint32_t value, uint32_t slot_generation) noexcept
	    : index(value), generation(slot_generation) {}

	[[nodiscard]] constexpr explicit operator bool() const noexcept { return index != INVALID_INDEX; }
	constexpr auto operator<=>(const SlotId&) const noexcept = default;

	uint32_t index = INVALID_INDEX;
	uint32_t generation = 0;
};

// Stable-address slot storage for cache resources that are intentionally non-movable.
//
// Layout: values live in fixed-size chunks that are never moved or freed until the vector is
// destroyed, so a reference obtained from operator[] stays valid across insert() (callers hold
// `auto& x = slots[id]` while inserting). The per-slot generation/liveness words live in a
// separate dense array. The validity check in operator[]/try_get/is_allocated therefore reads a
// small hot array instead of the tail of a large T (for Image the generation used to sit ~800
// bytes into the slot, a cache line the caller never touches otherwise, so every lookup paid an
// extra miss), and indexing is shift/mask plus two loads instead of std::deque's
// iterator arithmetic.
//
// The lookup path (operator[], try_get, is_allocated) is forced inline: it is a bounds check,
// two loads and a compare, and the texture cache calls it for every candidate image of every
// lookup. Left to the inliner, operator[] became an out-of-line call with a full frame for each
// T. The EXIT_IF failure branch is cold (see common/assert.h), so inlining adds no call sequence
// to the caller's hot path.
template <typename T>
class SlotVector {
public:
	SlotVector() = default;
	~SlotVector() {
		for (uint32_t index = 0; index < m_meta.size(); ++index) {
			if (m_meta[index].alive) {
				std::destroy_at(Value(index));
			}
		}
	}
	KYTY_CLASS_NO_COPY(SlotVector);

	[[nodiscard]] KYTY_FORCE_INLINE T& operator[](SlotId id) noexcept {
		EXIT_IF(!is_allocated(id));
		return *Value(id.index);
	}

	[[nodiscard]] KYTY_FORCE_INLINE const T& operator[](SlotId id) const noexcept {
		EXIT_IF(!is_allocated(id));
		return *Value(id.index);
	}

	[[nodiscard]] KYTY_FORCE_INLINE T* try_get(SlotId id) noexcept { return is_allocated(id) ? Value(id.index) : nullptr; }

	[[nodiscard]] KYTY_FORCE_INLINE const T* try_get(SlotId id) const noexcept {
		return is_allocated(id) ? Value(id.index) : nullptr;
	}

	[[nodiscard]] KYTY_FORCE_INLINE bool is_allocated(SlotId id) const noexcept {
		if (!id || id.index >= m_meta.size()) {
			return false;
		}
		const auto& meta = m_meta[id.index];
		return meta.alive && meta.generation == id.generation;
	}

	template <typename... Args>
	[[nodiscard]] SlotId insert(Args&&... args) {
		uint32_t index = 0;
		if (m_free_list.empty()) {
			index = static_cast<uint32_t>(m_meta.size());
			if ((index >> CHUNK_SHIFT) == m_chunks.size()) {
				// Raw storage, not value-initialised: no zero-fill of the whole chunk.
				m_chunks.push_back(std::make_unique_for_overwrite<Storage[]>(CHUNK_SIZE));
			}
			// The slot is published (not alive) before T is constructed, so a constructor that
			// re-enters insert() gets a different index.
			m_meta.emplace_back();
		} else {
			index = m_free_list.back();
			m_free_list.pop_back();
			EXIT_IF(m_meta[index].alive);
		}
		std::construct_at(Value(index), std::forward<Args>(args)...);
		m_meta[index].alive = true;
		++m_size;
		return SlotId {index, m_meta[index].generation};
	}

	void erase(SlotId id) noexcept {
		EXIT_IF(!is_allocated(id));
		std::destroy_at(Value(id.index));
		auto& meta = m_meta[id.index];
		meta.alive = false;
		if (++meta.generation == 0) {
			meta.generation = 1;
		}
		m_free_list.push_back(id.index);
		--m_size;
	}

	[[nodiscard]] size_t size() const noexcept { return m_size; }
	[[nodiscard]] size_t capacity() const noexcept { return m_meta.size(); }

	template <typename F>
	void ForEach(F&& fn) {
		for (uint32_t index = 0; index < m_meta.size(); ++index) {
			if (m_meta[index].alive) {
				fn(SlotId {index, m_meta[index].generation}, *Value(index));
			}
		}
	}

	template <typename F>
	void ForEach(F&& fn) const {
		for (uint32_t index = 0; index < m_meta.size(); ++index) {
			if (m_meta[index].alive) {
				fn(SlotId {index, m_meta[index].generation}, *Value(index));
			}
		}
	}

private:
	static constexpr uint32_t CHUNK_SHIFT = 6;
	static constexpr uint32_t CHUNK_SIZE  = 1u << CHUNK_SHIFT;
	static constexpr uint32_t CHUNK_MASK  = CHUNK_SIZE - 1;

	struct alignas(T) Storage {
		std::byte bytes[sizeof(T)];
	};

	struct SlotMeta {
		uint32_t generation = 1;
		bool     alive      = false;
	};

	[[nodiscard]] KYTY_FORCE_INLINE T* Value(uint32_t index) const noexcept {
		return std::launder(reinterpret_cast<T*>(m_chunks[index >> CHUNK_SHIFT][index & CHUNK_MASK].bytes));
	}

	std::vector<std::unique_ptr<Storage[]>> m_chunks;
	std::vector<SlotMeta>                   m_meta;
	std::vector<uint32_t>                   m_free_list;
	size_t                                  m_size = 0;
};

} // namespace Common

template <>
struct std::hash<Common::SlotId> {
	[[nodiscard]] size_t operator()(Common::SlotId id) const noexcept {
		return std::hash<uint64_t> {}((static_cast<uint64_t>(id.generation) << 32u) | id.index);
	}
};

#endif // EMULATOR_SRC_COMMON_SLOTVECTOR_H_
