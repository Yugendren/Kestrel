#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BLITHELPER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BLITHELPER_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <compare>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler;
class Image;
struct GraphicContext;

class BlitHelper final {
public:
	inline static constexpr auto ColorToMsDepthLayout =
	    vk::ImageLayout::eDepthStencilAttachmentOptimal;

	BlitHelper(GraphicContext& graphics, CommandScheduler& scheduler);
	~BlitHelper();
	KYTY_CLASS_NO_COPY(BlitHelper);

	void ReinterpretColorAsMsDepth(Image& source, Image& destination);

	// Resamples a depth surface onto one of a different size. vkCmdBlitImage is not a usable
	// transfer for the depth/stencil formats a render target uses here, so this redraws the
	// surface instead: a full-screen pass that point-samples the source and writes the
	// destination's depth, exporting stencil as well where the device can.
	void ResampleDepth(Image& source, Image& destination, const ImageSubresourceRange& source_range,
	                   const ImageSubresourceRange& destination_range);
	// Whether this surface can survive ResampleDepth intact.
	[[nodiscard]] static bool CanResampleDepth(const ImageInfo& info);

private:
	// The three passes differ only in what they write, so they share one cache.
	enum class PipelineKind : uint8_t { ColorToMsDepth, Depth, StencilBit };

	struct PipelineKey {
		PipelineKind kind    = PipelineKind::ColorToMsDepth;
		uint32_t     samples = 1;
		vk::Format   format  = vk::Format::eUndefined;

		auto operator<=>(const PipelineKey&) const = default;
	};

	struct Pipeline {
		PipelineKey  key;
		vk::Pipeline handle = nullptr;
	};

	[[nodiscard]] vk::ShaderModule CreateShader(const uint32_t* code, size_t words) const;
	[[nodiscard]] vk::Pipeline     GetPipeline(PipelineKey key);

	GraphicContext&         m_graphics;
	CommandScheduler&       m_scheduler;
	vk::DescriptorSetLayout m_descriptor_layout         = nullptr;
	vk::PipelineLayout      m_pipeline_layout           = nullptr;
	// The stencil pass selects its bit plane with a push constant, so it needs its own layout.
	vk::PipelineLayout      m_stencil_pipeline_layout   = nullptr;
	vk::ShaderModule        m_vertex_shader             = nullptr;
	vk::ShaderModule        m_fragment_shader           = nullptr;
	vk::ShaderModule        m_depth_shader              = nullptr;
	vk::ShaderModule        m_stencil_shader            = nullptr;
	std::vector<Pipeline>   m_pipelines;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BLITHELPER_H_
