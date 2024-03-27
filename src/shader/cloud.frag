#version 460

#define NB_STEPS 100

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

    //    return smallest_max > largest_min;
}

float getSample(float x, float y, float z) {
    return texture(volume, vec3(x, y, z)).r;
}

vec3
computeGradient(vec3 position, float step)
{
    return normalize(vec3(
                         getSample(position.x + step, position.y, position.z)
                         - getSample(position.x - step, position.y, position.z),
                         getSample(position.x, position.y + step, position.z)
                         - getSample(position.x, position.y - step, position.z),
                         getSample(position.x, position.y, position.z + step)
                         - getSample(position.x, position.y, position.z - step)
                     ));
}

void main() {
    Ray ray = frag_vray;
    ray.dir = normalize(frag_vray.dir);

    float near = 0.0;
    float far = 0.0;
    compute_near_far(ray, near, far);

    ray.origin = ray.origin + near * ray.dir;

    vec3 inc = 1.0 / abs(ray.dir);
    float delta = min(inc.x, min(inc.y, inc.z)) / float(NB_STEPS);
    ray.dir = ray.dir * delta;

//    vec3 base_color = vec3(0.4, 0.4, 0.5);
    vec3 base_color = vec3(0.4, 1.0, 0.4);
    vec4 acc = vec4(0.0);

    vec3 light_dir = vec3(0.0, -1.0, -1.0);

    float dist = near;

    for (int i = 0; i < NB_STEPS; ++i) {
        float s = texture(volume, ray.origin).r;

        //        vec3 gradient = computeGradient(ray.origin, delta);
        //        float NdotL = max(0., dot(gradient, light_dir));

        acc.rgb += (1.0 - acc.a) * s * base_color; //* NdotL;
        acc.a += (1.0 - acc.a) * s; //* 0.5;
        ray.origin += ray.dir;
        dist += delta;
        if (dist >= far) { break; }
        if (acc.a > 0.95) { break;}
    }

    out_color = acc;
}