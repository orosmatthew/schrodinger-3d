#version 460

layout (set = 1, binding = 1) uniform sampler2D tex_sampler;

layout (location = 0) in vec2 in_text_coord;
layout (location = 1) in vec3 in_text_color;

layout (location = 0) out vec4 out_color;

void main() {
    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(tex_sampler, in_text_coord).r);
    if (sampled.a < 0.0001) {
        discard;
    }
    out_color = vec4(in_text_color, 1.0) * sampled;
}