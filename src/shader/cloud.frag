#version 460

#define STEPS 100

struct Ray {
    vec3 origin;
    vec3 dir;
};

layout (set = 0, binding = 1) uniform sampler3D volume;

layout (location = 0) in Ray in_vray;
layout (location = 2) flat in int in_lighting;

layout (location = 0) out vec4 out_color;

vec3 light_dir = normalize(vec3(-1.0, -1.0, -1.0));

void compute_near_far(Ray ray, inout float near, inout float far) {
    vec3 inv_ray = 1.0 / ray.dir;

    // Shortcut here, it should be: `aabbMin - ray.origin`.
    // As we are always using normalized AABB, we can skip the line
    // `(0, 0, 0) - ray.origin`.
    vec3 tbottom = -inv_ray * ray.origin;
    vec3 ttop = inv_ray * (vec3(1.0) - ray.origin);

    vec3 tmin = min(ttop, tbottom);
    vec3 tmax = max(ttop, tbottom);

    float largest_min = max(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
    float smallest_max = min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));

    near = largest_min;
    far = smallest_max;
}

void main() {
    Ray ray = in_vray;
    ray.dir = normalize(in_vray.dir);

    float near = 0.0;
    float far = 0.0;
    compute_near_far(ray, near, far);

    near = max(0.0, near);
    ray.origin = ray.origin + near * ray.dir;

    vec3 inc = 1.0 / abs(ray.dir);
    float delta = min(inc.x, min(inc.y, inc.z)) / float(STEPS);
    ray.dir = ray.dir * delta;

    vec3 base_color = vec3(0.4, 1.0, 0.4);
    vec4 acc = vec4(0.0);

    float dist = near;

    for (int i = 0; i < STEPS; ++i) {
        float s = texture(volume, ray.origin).a;

        float lighting = 1.0;
        if (in_lighting == 1) {
            // Calculate the gradient (normal) of the smoke
            vec3 gradient = vec3(
            texture(volume, ray.origin + vec3(0.01, 0, 0)).a - texture(volume, ray.origin - vec3(0.01, 0, 0)).a,
            texture(volume, ray.origin + vec3(0, 0.01, 0)).a - texture(volume, ray.origin - vec3(0, 0.01, 0)).a,
            texture(volume, ray.origin + vec3(0, 0, 0.01)).a - texture(volume, ray.origin - vec3(0, 0, 0.01)).a
            );
            gradient = normalize(gradient);

            // Calculate the lighting as the dot product of the gradient and the light direction
            lighting = max(0.2, dot(gradient, light_dir));
        }

        acc.rgb += (1.0 - acc.a) * s * texture(volume, ray.origin).rgb * lighting;
        acc.a += (1.0 - acc.a) * s;

        ray.origin += ray.dir;
        dist += delta;
        if (dist >= far) { break; }
        if (acc.a > 1.0) { break; }
    }

    out_color = acc;
}
