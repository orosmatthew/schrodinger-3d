#version 460

layout (set = 0, binding = 0) uniform GlobalUniform {
    mat4 view;
    mat4 proj;
} global_ubo;

layout (set = 1, binding = 0) uniform ObjectUnifom {
    mat4 model;
} object_ubo;

layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec3 in_color;

layout (location = 0) out vec3 frag_color;

void main() {
    gl_Position = global_ubo.proj * global_ubo.view * object_ubo.model * vec4(in_pos, 1.0);

    frag_color = in_color;
}