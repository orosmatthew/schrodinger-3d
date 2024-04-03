#include <chrono>

#include "mve/math/math.hpp"
#include "mve/renderer.hpp"
#include "mve/shader.hpp"
#include "mve/vertex_data.hpp"
#include "mve/window.hpp"

#include "camera.hpp"
#include "logger.hpp"
#include "schrodinger_sim_3d.hpp"
#include "simple_pipeline.hpp"
#include "text_buffer.hpp"
#include "text_pipeline.hpp"
#include "util/fixed_loop.hpp"
#include "wire_box_mesh.hpp"

void fill_buffer_sim(const SchrodingerSim3d& sim, std::vector<std::byte>& buffer)
{
    double prob_min = std::numeric_limits<double>::max();
    double prob_max = std::numeric_limits<double>::min();
    for (int i = 0; i < sim.size() * sim.size() * sim.size(); ++i) {
        const auto sim_value = sim.value_at_idx(i);
        const auto abs = std::norm(sim_value);
        if (abs > prob_max) {
            prob_max = abs;
        }
        if (abs < prob_min) {
            prob_min = abs;
        }
    }
    for (int i = 0; i < sim.size() * sim.size() * sim.size(); ++i) {
        const float sim_value = static_cast<float>(std::norm(sim.value_at_idx(i)));
        buffer[i] = static_cast<std::byte>(std::clamp((sim_value - prob_min) / prob_max, 0.0, 1.0) * 255);
    }
}

struct SimMeshData {
    mve::VertexLayout vertex_layout;
    mve::VertexBuffer vertex_buffer;
    mve::IndexBuffer index_buffer;
};

void init_packet(SchrodingerSim3d& sim)
{
    sim.lock_write();
    constexpr auto i = std::complex(0.0, 1.0);
    for (int j = 0; j < sim.size() * sim.size() * sim.size(); ++j) {
        constexpr auto a = 1.0;
        constexpr auto x0 = 10;
        constexpr auto y0 = 32;
        constexpr auto z0 = 32;
        constexpr auto sigma_x = 10.0;
        constexpr auto sigma_y = 10.0;
        constexpr auto sigma_z = 2.0;
        constexpr auto mom_x = 2.0;
        constexpr auto mom_y = 0.0;
        constexpr auto mom_z = 0.0;
        const auto [x, y, z] = sim.idx_to_pos(j);
        const auto x_term = std::exp(-std::pow(x - x0, 2.0) / (2.0 * std::pow(sigma_x, 2.0)));
        const auto y_term = std::exp(-std::pow(y - y0, 2.0) / (2.0 * std::pow(sigma_y, 2.0)));
        const auto z_term = std::exp(-std::pow(z - z0, 2.0) / (2.0 * std::pow(sigma_z, 2.0)));
        const auto pos = x_term * y_term * z_term;
        const auto mom = std::exp(i * (mom_x * x + mom_y * y + mom_z * z));
        sim.set_at({ x, y, z }, a * pos * mom);
    }
    sim.unlock_write();
}

SimMeshData create_sim_mesh_data(mve::Renderer& renderer)
{
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

    const std::array<uint32_t, 6> base_indices = { 0, 3, 2, 0, 2, 1 };
    std::vector<uint32_t> indices;
    indices.reserve(6 * 6);
    for (int f = 0; f < 6; f++) {
        for (int i = 0; i < 6; i++) {
            indices.push_back(base_indices[i] + f * 4);
        }
    }
    return SimMeshData { .vertex_layout = std::move(vertex_layout),
                         .vertex_buffer = renderer.create_vertex_buffer(data),
                         .index_buffer = renderer.create_index_buffer(indices) };
}

struct GlobalData {
    std::thread sim_thread;
    std::atomic<bool> should_exit = false;
    std::unique_ptr<SchrodingerSim3d> sim;
    std::atomic<int> sim_frame_count = 0;
};
static GlobalData g_global_data;

void sim_thread_process()
{
    while (!g_global_data.should_exit) {
        g_global_data.sim->update();
        ++g_global_data.sim_frame_count;
    }
}

int main()
{
    initLogger();

    mve::Window window("Schrodinger Sim 3D", mve::Vector2i(600, 600), true);
    mve::Renderer renderer(window, "Schrodinger Sim 3D", 1, 0, 0);

    auto [vertex_layout, vertex_buffer, index_buffer] = create_sim_mesh_data(renderer);
    mve::Shader vert_shader("../res/bin/shader/cloud.vert.spv");
    mve::Shader frag_shader("../res/bin/shader/cloud.frag.spv");
    mve::GraphicsPipeline pipeline = renderer.create_graphics_pipeline(
        vert_shader, frag_shader, vertex_layout, mve::CullMode::back, mve::DepthTest::on);

    mve::DescriptorSet global_descriptor = pipeline.create_descriptor_set(vert_shader.descriptor_set(0));
    mve::UniformBuffer global_ubo = renderer.create_uniform_buffer(vert_shader.descriptor_set(0).binding(0));
    global_descriptor.write_binding(vert_shader.descriptor_set(0).binding(0), global_ubo);
    mve::Matrix4 proj = mve::perspective(90.0f, 1.0f, 0.001f, 100.0f);
    global_ubo.update(vert_shader.descriptor_set(0).binding(0).member("proj").location(), proj);

    Camera camera;

    SchrodingerSim3d::Properties sim_props {
        .size = 64, .grid_spacing = 1.0, .timestep = 0.004, .hbar = 1.0, .mass = 1.0
    };
    g_global_data.sim = std::make_unique<SchrodingerSim3d>(sim_props);

    init_packet(*g_global_data.sim);

    std::vector<std::byte> buffer(sim_props.size * sim_props.size * sim_props.size);

    mve::Texture texture = renderer.create_texture(mve::TextureFormat::r, mve::Vector3i(sim_props.size), buffer.data());

    global_descriptor.write_binding(frag_shader.descriptor_set(0).binding(1), texture);

    util::FixedLoop fixed_loop(60.0f);

    TextPipeline text_pipeline(renderer, 36);
    TextBuffer fps_text = text_pipeline.create_text_buffer("FPS:", { 0.0f, 0.0f }, 1.0f, { 1.0f, 1.0f, 1.0f });
    TextBuffer tick_text = text_pipeline.create_text_buffer("TPS: ", { 0.0f, 40.0f }, 1.0f, { 1.0f, 1.0f, 1.0f });

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
        0.002f,
        { 0.2f, 0.2f, 0.2f });

    g_global_data.sim_thread = std::thread(sim_thread_process);

    window.disable_cursor();
    bool cursor_captured = true;
    int current_frame_count = 0;
    int frame_count;
    auto begin_time = std::chrono::steady_clock::time_point();
    while (!window.should_close() && !g_global_data.should_exit) {
        window.poll_events();

        if (window.is_key_down(mve::Key::left_alt) && window.is_key_pressed(mve::Key::enter)) {
            if (window.is_fullscreen()) {
                window.windowed();
            }
            else {
                window.fullscreen(true);
            }
        }

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

        if (window.is_key_pressed(mve::Key::escape)) {
            g_global_data.should_exit = true;
        }

        camera.update(window);
        fixed_loop.update(1, [&] { camera.fixed_update(window); });

        g_global_data.sim->lock_read();
        fill_buffer_sim(*g_global_data.sim, buffer);
        g_global_data.sim->unlock_read();
        texture.update(buffer.data());

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
        text_pipeline.draw(tick_text);

        renderer.end_render_pass();
        renderer.end_frame(window);

        if (std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count() >= 1000000) {
            begin_time = std::chrono::steady_clock::now();
            frame_count = current_frame_count;
            fps_text.update("FPS:" + std::to_string(frame_count));
            tick_text.update("TPS:" + std::to_string(g_global_data.sim_frame_count));
            g_global_data.sim_frame_count = 0;
            current_frame_count = 0;
        }
        current_frame_count++;
    }
    g_global_data.sim_thread.join();
}