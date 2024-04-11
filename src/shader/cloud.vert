#version 460

layout (set = 0, binding = 0) uniform GlobalUniform {
    mat4 view;
    mat4 proj;
    int lighting;
} global_ubo;

struct Ray {
    vec3 origin;
    vec3 dir;
};

layout (location = 0) in vec3 in_pos;

layout (location = 0) out Ray out_vray;
layout (location = 2) flat out int out_lighting;

void main() {
    vec3 cam_pos = (inverse(global_ubo.view) * vec4(0.0, 0.0, 0.0, 1.0)).xyz;

    out_vray.dir = in_pos - cam_pos;
    out_vray.origin = cam_pos + vec3(0.5);
    out_lighting = global_ubo.lighting;

    gl_Position = global_ubo.proj * global_ubo.view * vec4(in_pos, 1.0);
}