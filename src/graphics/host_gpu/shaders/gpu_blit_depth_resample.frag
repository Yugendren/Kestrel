#version 450 core
#extension GL_EXT_samplerless_texture_functions : require

// Point-samples one depth surface onto another of a different size. Depth is never filtered:
// an interpolated depth value describes a surface that was never drawn.
layout(binding = 0, set = 0) uniform texture2D depth_source;
layout(location = 0) in vec2 uv;

void main() {
	ivec2 size   = textureSize(depth_source, 0);
	ivec2 coord  = clamp(ivec2(uv * vec2(size)), ivec2(0), size - ivec2(1));
	gl_FragDepth = texelFetch(depth_source, coord, 0).x;
}
