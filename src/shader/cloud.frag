#version 460

#define STEPS 200

struct Ray {
    vec3 origin;
    vec3 dir;
};

layout (set = 0, binding = 1) uniform sampler3D volume;

layout (location = 0) in Ray frag_vray;

layout (location = 0) out vec4 out_color;

void compute_near_far(Ray ray, inout float near, inout float far) {
    // Ray is assumed to be in local coordinates, ie:
    // ray = inverse(objectMatrix * invCameraMatrix) * ray
    // Equation of ray: O + D * t

    vec3 invRay = 1.0 / ray.dir;

    // Shortcut here, it should be: `aabbMin - ray.origin`.
    // As we are always using normalized AABB, we can skip the line
    // `(0, 0, 0) - ray.origin`.
    vec3 tbottom = -invRay * ray.origin;
    vec3 ttop = invRay * (vec3(1.0) - ray.origin);

    vec3 tmin = min(ttop, tbottom);
    vec3 tmax = max(ttop, tbottom);

    float largestMin = max(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
    float smallestMax = min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));

    near = largestMin;
    far = smallestMax;
}

void main() {
    Ray ray = frag_vray;
    ray.dir = normalize(frag_vray.dir);

    float near = 0.0;
    float far = 0.0;
    compute_near_far(ray, near, far);

    if (near < 0.0) near = 0.0;
    ray.origin = ray.origin + near * ray.dir;

    vec3 inc = 1.0 / abs(ray.dir);
    float delta = min(inc.x, min(inc.y, inc.z)) / float(STEPS);
    ray.dir = ray.dir * delta;

    vec3 base_color = vec3(0.4, 1.0, 0.4);
    vec4 acc = vec4(0.0);

    float dist = near;

    for (int i = 0; i < STEPS; ++i) {
        float s = texture(volume, ray.origin).r;
        /*
        if (s > 0.0) {
            acc.rgb =  base_color;
            acc.a = 1.0; //* 0.5;
            break;
        }
        */
        acc.rgb += (1.0 - acc.a) * s * base_color;
        acc.a += (1.0 - acc.a) * s; //* 0.5;
        ray.origin += ray.dir;
        dist += delta;
        if (dist >= far) { break; }
        if (acc.a > 1.0) { break;}
    }

    out_color = acc;
}