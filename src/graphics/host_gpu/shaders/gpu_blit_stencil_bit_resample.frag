#version 450 core
#extension GL_EXT_samplerless_texture_functions : require

// Copies one bit plane of a stencil surface onto another of a different size. A fragment shader
// cannot write the stencil aspect, so the pass instead keeps only the fragments whose source bit
// is set and lets the pipeline replace that one bit in the destination. Eight of these rebuild
// the whole value, which is what stencil export would have done in a single pass.
layout(binding = 0, set = 0) uniform utexture2D stencil_source;
layout(push_constant) uniform Constants {
	uint bit;
} constants;
layout(location = 0) in vec2 uv;

void main() {
	ivec2 size  = textureSize(stencil_source, 0);
	ivec2 coord = clamp(ivec2(uv * vec2(size)), ivec2(0), size - ivec2(1));
	if ((texelFetch(stencil_source, coord, 0).x & (1u << constants.bit)) == 0u) {
		discard;
	}
}
