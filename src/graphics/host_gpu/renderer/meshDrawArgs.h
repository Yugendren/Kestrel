#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHDRAWARGS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHDRAWARGS_H_

#include "common/abi.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {

struct GraphicContext;

// Converts a guest DrawIndexedIndirectArgs block into the six-dword mesh draw parameter block a
// mesh program addresses through push constants, plus a VkDrawMeshTasksIndirectCommandEXT sized
// from it. Doing this conversion on the GPU is what lets an indirect mesh draw stay there too:
// without it the six dwords could only be filled by a host that has already read the guest's
// counts back, which is the stall an indirect draw exists to avoid. One instance is shared by
// every mesh draw the renderer records.
class MeshDrawArgsBuilder {
public:
	// The sizes of the two buffer ranges Record() writes, in dwords: the mesh draw parameter block
	// (index_count, vertex_offset, first_instance, element_size, index address low/high) and a
	// VkDrawMeshTasksIndirectCommandEXT.
	static constexpr uint32_t ParamsDwordCount   = 6;
	static constexpr uint32_t DispatchDwordCount = 3;

	explicit MeshDrawArgsBuilder(GraphicContext& graphics);
	~MeshDrawArgsBuilder();
	KYTY_CLASS_NO_COPY(MeshDrawArgsBuilder);

	struct Args {
		// The guest's DrawIndexedIndirectArgs record, at its exact (unaligned) byte offset; Record()
		// floors it to a descriptor-legal offset itself and carries the remainder in a push constant.
		vk::Buffer     guest_args_buffer = nullptr;
		vk::DeviceSize guest_args_offset = 0;
		// Destinations for the two outputs. Both offsets must already satisfy
		// GraphicContext::StorageMinAlignment(): unlike the guest args offset above, the caller
		// chooses them and can allocate accordingly.
		vk::Buffer     params_buffer     = nullptr;
		vk::DeviceSize params_offset     = 0;
		vk::Buffer     dispatch_buffer   = nullptr;
		vk::DeviceSize dispatch_offset   = 0;
		// How index_count becomes a primitive, then a workgroup, count -- the same quantities
		// ShaderMeshInputInfo::InputPrimitiveSize/Step() and primitives_per_group give the host for a
		// direct mesh draw.
		uint32_t primitive_size      = 0;
		uint32_t primitive_step      = 0;
		uint32_t primitives_per_group = 0;
		// Guest index element size in bytes and the guest base address of the index buffer; the
		// shader adds start_index * element_size to the latter itself.
		uint32_t element_size = 0;
		uint64_t index_base   = 0;
		// mesh_shader_properties.maxMeshWorkGroupCount[0]/[1] and maxMeshWorkGroupTotalCount: a
		// direct draw has the host EXIT if it would exceed these, but an indirect draw's counts are
		// not known until the shader runs, so it clamps to them instead (y last, so x * y stays
		// within the total).
		uint32_t max_groups_x     = 0;
		uint32_t max_groups_y     = 0;
		uint32_t max_groups_total = 0;
	};

	// Records the conversion: a compute dispatch that reads `args` and writes the parameter and
	// dispatch buffers. Must be recorded outside a render pass instance, like any other compute
	// work, and the caller barriers both destination ranges before anything reads them.
	void Record(vk::CommandBuffer command, const Args& args) const;

private:
	GraphicContext&         m_graphics;
	vk::DescriptorSetLayout m_desc_layout      = nullptr;
	vk::PipelineLayout      m_pipeline_layout  = nullptr;
	vk::Pipeline            m_pipeline         = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHDRAWARGS_H_
