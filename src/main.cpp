#include <chrono>

#include <BS_thread_pool.hpp>
#include <FastNoiseLite.h>

#include "mve/detail/defs.hpp"
#include "mve/math/math.hpp"
#include "mve/renderer.hpp"
#include "mve/shader.hpp"
#include "mve/vertex_data.hpp"
#include "mve/window.hpp"

#include "camera.hpp"
#include "fluid3d.hpp"
#include "logger.hpp"
#include "simple_pipeline.hpp"
#include "text_buffer.hpp"
#include "text_pipeline.hpp"
#include "util/fixed_loop.hpp"
#include "wire_box_mesh.hpp"

void fill_buffer(
    FastNoiseLite& noise,
    std::vector<std::byte>& buffer,
    const int width,
    const int height,
    const int depth,
    const float scale,
    const mve::Vector3 noise_offset)
{
    const int voxel_per_slice = width * height;
    const int voxel_count = voxel_per_slice * depth;

    noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2S);

    for (int i = 0; i < voxel_count; ++i) {
        const int x = i % width;
        const auto y = (i % voxel_per_slice) / width;
        const auto z = i / voxel_per_slice;
        const float p = noise.GetNoise(
            static_cast<float>(x) * scale + noise_offset.x,
            static_cast<float>(y) * scale + noise_offset.y,
            static_cast<float>(z) * scale + noise_offset.z);
        const float rand = (p + 1.0f) * 0.5f;
        buffer[i] = static_cast<std::byte>(mve::clamp(static_cast<int>(mve::round(rand * 16)), 0, 16));
    }
}

void fill_buffer_fluid(
    const Fluid3D& fluid, std::vector<std::byte>& buffer, const int width, const int height, const int depth)
{
    const int voxel_per_slice = width * height;
    const int voxel_count = voxel_per_slice * depth;

    for (int i = 0; i < voxel_count; ++i) {
        const int x = i % width;
        const auto y = (i % voxel_per_slice) / width;
        const auto z = i / voxel_per_slice;
        const float p = fluid.density_at(x, y, z);
        buffer[i] = static_cast<std::byte>(mve::clamp(static_cast<int>(p), 0, 255));
    }
}

int main()
{
    initLogger();

    mve::Window window("Fluid Sim 3D", mve::Vector2i(600, 600), true);
    mve::Renderer renderer(window, "Fluid Sim 3D", 1, 0, 0);

    mve::VertexLayout vertex_layout;
    vertex_layout.push_back(mve::VertexAttributeType::vec3); // pos

    mve::VertexData data(vertex_layout);
    data.push_back(mve::Vector3(-0.5f, -0.5f, 0.5f));
    data.push_back(mve::Vector3(0.5f, -0.5f, 0.5f));
    data.push_back(mve::Vector3(0.5f, -0.5f, -0.5f));
    data.push_back(mve::Vector3(-0.5f, -0.5f, -0.5f));

    data.push_back(mve::Vector3(0.5f, 0.5f, 0.5f));
    data.push_back(mve::Vector3(-0.5f, 0.5f, 0.5f));
    data.push_back(mve::Vector3(-0.5f, 0.5f, -0.5f));
    data.push_back(mve::Vector3(0.5f, 0.5f, -0.5f));

    data.push_back(mve::Vector3(-0.5f, 0.5f, 0.5f));
    data.push_back(mve::Vector3(-0.5f, -0.5f, 0.5f));
    data.push_back(mve::Vector3(-0.5f, -0.5f, -0.5f));
    data.push_back(mve::Vector3(-0.5f, 0.5f, -0.5f));

    data.push_back(mve::Vector3(0.5f, -0.5f, 0.5f));
    data.push_back(mve::Vector3(0.5f, 0.5f, 0.5f));
    data.push_back(mve::Vector3(0.5f, 0.5f, -0.5f));
    data.push_back(mve::Vector3(0.5f, -0.5f, -0.5f));

    data.push_back(mve::Vector3(-0.5f, 0.5f, 0.5f));
    data.push_back(mve::Vector3(0.5f, 0.5f, 0.5f));
    data.push_back(mve::Vector3(0.5f, -0.5f, 0.5f));
    data.push_back(mve::Vector3(-0.5f, -0.5f, 0.5f));

    data.push_back(mve::Vector3(0.5f, 0.5f, -0.5f));
    data.push_back(mve::Vector3(-0.5f, 0.5f, -0.5f));
    data.push_back(mve::Vector3(-0.5f, -0.5f, -0.5f));
    data.push_back(mve::Vector3(0.5f, -0.5f, -0.5f));

    std::array<uint32_t, 6> base_indices = { 0, 3, 2, 0, 2, 1 };
    std::vector<uint32_t> indices;
    indices.reserve(6 * 6);
    for (int f = 0; f < 6; f++) {
        for (int i = 0; i < 6; i++) {
            indices.push_back(base_indices[i] + (f * 4));
        }
    }

    mve::VertexBuffer vertex_buffer = renderer.create_vertex_buffer(data);
    mve::IndexBuffer index_buffer = renderer.create_index_buffer(indices);

    mve::Shader vert_shader("../res/bin/shader/cloud.vert.spv");
    mve::Shader frag_shader("../res/bin/shader/cloud.frag.spv");
    mve::GraphicsPipeline pipeline = renderer.create_graphics_pipeline(vert_shader, frag_shader, vertex_layout, true);

    mve::DescriptorSet global_descriptor = pipeline.create_descriptor_set(vert_shader.descriptor_set(0));
    mve::UniformBuffer global_ubo = renderer.create_uniform_buffer(vert_shader.descriptor_set(0).binding(0));
    global_descriptor.write_binding(vert_shader.descriptor_set(0).binding(0), global_ubo);
    mve::Matrix4 proj = mve::perspective(90.0f, 1.0f, 0.001f, 100.0f);
    global_ubo.update(vert_shader.descriptor_set(0).binding(0).member("proj").location(), proj);

    Camera camera;

    // Clouds Volume
    constexpr int width = 72;
    constexpr int height = 72;
    constexpr int depth = 72;
    constexpr float scale = 4.0f;
    std::vector<std::byte> buffer(width * height * depth);

    FastNoiseLite noise;
    noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2S);
    fill_buffer(noise, buffer, width, height, depth, scale, { 0, 0, 0 });

    mve::Texture texture = renderer.create_texture(mve::TextureFormat::r, width, height, depth, buffer.data());

    global_descriptor.write_binding(frag_shader.descriptor_set(0).binding(1), texture);

    util::FixedLoop fixed_loop(60.0f);

    TextPipeline text_pipeline(renderer, 36);
    TextBuffer fps_text = text_pipeline.create_text_buffer("FPS:", { 0, 0 }, 1.0f, { 1.0f, 1.0f, 1.0f });

    SimplePipeline simple_pipeline(renderer);

    window.set_resize_callback([&](const mve::Vector2i new_size) {
        renderer.resize(window);
        text_pipeline.resize();
        simple_pipeline.resize(new_size);
        const mve::Matrix4 new_proj
            = mve::perspective(90.0f, static_cast<float>(new_size.x) / static_cast<float>(new_size.y), 0.001f, 100.0f);
        global_ubo.update(vert_shader.descriptor_set(0).binding(0).member("proj").location(), new_proj);
    });
    text_pipeline.resize();
    simple_pipeline.resize({ 600, 600 });

    BoundingBox bb = { { -0.5f, -0.5f, -0.5f }, { 0.5f, 0.5f, 0.5f } };
    WireBoxMesh box(
        renderer,
        simple_pipeline.pipeline(),
        simple_pipeline.model_descriptor_set(),
        simple_pipeline.model_uniform_binding(),
        bb,
        0.01f,
        { 1.0f, 1.0f, 1.0f });

    Fluid3D fluid(72, 0.0f, 0.0f, 4);

    BS::thread_pool thread_pool;

    window.disable_cursor();
    bool cursor_captured = true;
    int current_frame_count = 0;
    int frame_count;
    auto begin_time = std::chrono::steady_clock::time_point();
    while (!window.should_close()) {
        window.poll_events();

        if (window.is_key_pressed(mve::Key::c)) {
            if (cursor_captured) {
                window.enable_cursor();
                cursor_captured = false;
            }
            else {
                window.disable_cursor();
                cursor_captured = true;
            }
        }

        if (window.is_key_pressed(mve::Key::f)) {
            if (window.is_fullscreen()) {
                window.windowed();
            }
            else {
                window.fullscreen(true);
            }
        }

        if (window.is_key_pressed(mve::Key::escape)) {
            break;
        }

        if (window.is_mouse_button_down(mve::MouseButton::left)) {
            fluid.add_density(32, 32, 32, 1000.0f);
            // mve::Vector3 dir = camera.look_direction();
            // LOG->info("DIR: ({}, {}, {})", dir.x, dir.y, dir.z);
            fluid.add_velocity(32, 32, 32, 10.0f, 10.0f, 0.0f);
        }

        camera.update(window);
        fixed_loop.update(1, [&] {
            camera.fixed_update(window);
            thread_pool.wait_for_tasks();
            fill_buffer_fluid(fluid, buffer, width, height, depth);
            texture.update(buffer.data());
            thread_pool.push_task([&] { fluid.step(1.0f / 60.0f); });
        });

        global_ubo.update(
            vert_shader.descriptor_set(0).binding(0).member("view").location(), camera.view_matrix(fixed_loop.blend()));
        simple_pipeline.set_view(camera.view_matrix(fixed_loop.blend()));

        renderer.begin_frame(window);
        renderer.begin_render_pass_present();

        renderer.bind_graphics_pipeline(simple_pipeline.pipeline());
        box.draw(simple_pipeline.global_descriptor_set());

        renderer.bind_graphics_pipeline(pipeline);
        renderer.bind_descriptor_set(global_descriptor);
        renderer.bind_vertex_buffer(vertex_buffer);
        renderer.draw_index_buffer(index_buffer);

        text_pipeline.draw(fps_text);

        renderer.end_render_pass_present();
        renderer.end_frame(window);

        if (std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count() >= 1000000) {
            begin_time = std::chrono::high_resolution_clock::now();
            frame_count = current_frame_count;
            fps_text.update("FPS:" + std::to_string(frame_count));
            current_frame_count = 0;
        }
        current_frame_count++;
    }
}