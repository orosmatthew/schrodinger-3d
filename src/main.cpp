#include <chrono>

#include "mve/math/math.hpp"
#include "mve/renderer.hpp"
#include "mve/shader.hpp"
#include "mve/vertex_data.hpp"
#include "mve/window.hpp"

#include "camera.hpp"
#include "logger.hpp"
#include "scenarios.hpp"
#include "schrodinger_sim_3d.hpp"
#include "simple_pipeline.hpp"
#include "text_buffer.hpp"
#include "text_pipeline.hpp"
#include "util/fixed_loop.hpp"
#include "wire_box_mesh.hpp"

struct SimMeshData {
    mve::VertexLayout vertex_layout;
    mve::VertexBuffer vertex_buffer;
    mve::IndexBuffer index_buffer;
};

struct GlobalData {
    std::thread sim_thread;
    std::atomic<bool> paused = false;
    std::atomic<bool> should_exit = false;
    std::unique_ptr<SchrodingerSim3d> sim;
    std::atomic<int> sim_frame_count = 0;
};

enum class SimDisplayTheme { probability, components };

enum class Scenario { simple_packet, double_slit, double_slit_potential, wall_potential, moving_sphere_potential };

static BS::thread_pool g_thread_pool;
static GlobalData g_global_data;

static std::string fps_display_text(const int fps)
{
    return "FPS: " + std::to_string(fps);
}

static std::string tps_display_text(const int tps)
{
    return "TPS: " + std::to_string(tps);
}

static std::string theme_display_text(const SimDisplayTheme theme)
{
    switch (theme) {
    case SimDisplayTheme::probability:
        return "[t] Theme: probability";
    case SimDisplayTheme::components:
        return "[t] Theme: components";
    }
    return "[t] Theme: unknown";
}

static std::string cursor_display_text(const bool captured)
{
    if (captured) {
        return "[c] Cursor: captured";
    }
    return "[c] Cursor: free";
}

static std::string lighting_display_text(const bool lighting)
{
    if (lighting) {
        return "[l] Lighting: on";
    }
    return "[l] Lighting: off";
}

static std::string fullscreen_display_text(const bool fullscreen)
{
    if (fullscreen) {
        return "[f] Fullscreen: on";
    }
    return "[f] Fullscreen: off";
}

static void fill_buffer_sim(const SchrodingerSim3d& sim, std::vector<std::byte>& buffer, const SimDisplayTheme theme)
{
    struct Extremes {
        double prob_min = std::numeric_limits<double>::max();
        double prob_max = std::numeric_limits<double>::min();
        double pot_min = std::numeric_limits<double>::max();
        double pot_max = std::numeric_limits<double>::min();
    };
    Extremes extremes;
    for (BS::multi_future<Extremes> extreme_futures = g_thread_pool.submit_blocks(
             0,
             sim.size() * sim.size() * sim.size(),
             [&sim](const int start, const int end) {
                 double block_min = std::numeric_limits<double>::max();
                 double block_max = std::numeric_limits<double>::min();
                 double block_pot_min = std::numeric_limits<double>::max();
                 double block_pot_max = std::numeric_limits<double>::min();
                 for (int i = start; i < end; ++i) {
                     const auto sim_value = sim.value_at_idx(i);
                     const auto abs = std::norm(sim_value);
                     block_min = std::min(abs, block_min);
                     block_max = std::max(abs, block_max);
                     const double pot = sim.potential_at_idx(i);
                     block_pot_min = std::min(pot, block_pot_min);
                     block_pot_max = std::max(pot, block_pot_max);
                 }
                 return Extremes {
                     .prob_min = block_min, .prob_max = block_max, .pot_min = block_pot_min, .pot_max = block_pot_max
                 };
             });
         std::future<Extremes> & future : extreme_futures) {
        auto [prob_min, prob_max, pot_min, pot_max] = future.get();
        extremes.prob_min = std::min(extremes.prob_min, prob_min);
        extremes.prob_max = std::max(extremes.prob_max, prob_max);
        extremes.pot_min = std::min(extremes.pot_min, pot_min);
        extremes.pot_max = std::max(extremes.pot_max, pot_max);
    }
    const bool draw_potential = !(extremes.pot_min == 0 && extremes.pot_max == 0);
    g_thread_pool.detach_blocks(0, sim.size() * sim.size() * sim.size(), [&](const int start, const int end) {
        for (int i = start; i < end; ++i) {
            const int base = i * 4;
            if (sim.fixed_at_idx(i)) {
                buffer[base] = static_cast<std::byte>(0);
                buffer[base + 1] = static_cast<std::byte>(0);
                buffer[base + 2] = static_cast<std::byte>(255);
                buffer[base + 3] = static_cast<std::byte>(255);
                continue;
            }
            if (theme == SimDisplayTheme::components) {
                constexpr double comp_min = 0;
                constexpr double comp_max = 0.0001;
                const std::complex<double> sim_value = sim.value_at_idx(i);
                const auto real = sim_value.real();
                const auto imag = sim_value.imag();
                buffer[base] = static_cast<std::byte>(std::clamp((real - comp_min) / comp_max, 0.0, 1.0) * 255);
                buffer[base + 1] = static_cast<std::byte>(std::clamp((imag - comp_min) / comp_max, 0.0, 1.0) * 255);
                buffer[base + 2] = static_cast<std::byte>(0);
                buffer[base + 3] = static_cast<std::byte>(
                    std::clamp((std::norm(sim_value) - extremes.prob_min) / extremes.prob_max, 0.0, 1.0) * 255);
            }
            else {
                const double sim_value = std::norm(sim.value_at_idx(i));
                buffer[base] = static_cast<std::byte>(255);
                buffer[base + 1] = static_cast<std::byte>(255);
                buffer[base + 2] = static_cast<std::byte>(255);
                buffer[base + 3] = static_cast<std::byte>(
                    std::clamp((sim_value - extremes.prob_min) / extremes.prob_max, 0.0, 1.0) * 255);
            }
            if (draw_potential) {
                if (const auto pot = sim.potential_at_idx(i); pot != 0) {
                    buffer[base] = static_cast<std::byte>(255);
                    buffer[base + 1] = static_cast<std::byte>(0);
                    buffer[base + 2] = static_cast<std::byte>(255);
                    buffer[base + 3] = static_cast<std::byte>(
                        std::clamp((pot - extremes.pot_min) / extremes.pot_max, 0.0, 1.0) * 255);
                }
            }
        }
    });
    g_thread_pool.wait();
}

static SimMeshData create_sim_mesh_data(mve::Renderer& renderer)
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

static void sim_thread_process()
{
    while (!g_global_data.should_exit) {
        if (g_global_data.paused) {
            std::this_thread::yield();
        }
        else {
            g_global_data.sim->update();
            ++g_global_data.sim_frame_count;
        }
    }
}

int main()
{
    initLogger();

    const mve::Vector2i init_screen_size(800, 600);
    mve::Window window("Schrodinger Sim 3D", init_screen_size, true);
    mve::Renderer renderer(window, "Schrodinger Sim 3D", 1, 0, 0);

    auto [vertex_layout, vertex_buffer, index_buffer] = create_sim_mesh_data(renderer);
    mve::Shader vert_shader("./res/bin/shader/cloud.vert.spv");
    mve::Shader frag_shader("./res/bin/shader/cloud.frag.spv");
    mve::GraphicsPipeline pipeline = renderer.create_graphics_pipeline(
        vert_shader, frag_shader, vertex_layout, mve::CullMode::front, mve::DepthTest::on);

    mve::DescriptorSet global_descriptor = pipeline.create_descriptor_set(vert_shader.descriptor_set(0));
    mve::UniformBuffer global_ubo = renderer.create_uniform_buffer(vert_shader.descriptor_set(0).binding(0));
    global_descriptor.write_binding(vert_shader.descriptor_set(0).binding(0), global_ubo);
    mve::Matrix4 proj = mve::perspective(90.0f, init_screen_size.aspect_ratio(), 0.001f, 100.0f);
    global_ubo.update(vert_shader.descriptor_set(0).binding(0).member("proj").location(), proj);
    const mve::UniformLocation shader_lighting_location
        = vert_shader.descriptor_set(0).binding(0).member("lighting").location();
    bool shader_lighting = true;
    global_ubo.update(shader_lighting_location, shader_lighting);

    Camera camera;

    SchrodingerSim3d::Properties sim_props {
        .size = 64, .grid_spacing = 1.0, .timestep = 0.002, .hbar = 1.0, .mass = 1.0
    };
    g_global_data.sim = std::make_unique<SchrodingerSim3d>(sim_props);

    scenario_simple_packet(*g_global_data.sim);

    std::vector<std::byte> buffer(sim_props.size * sim_props.size * sim_props.size * 4);

    mve::Texture texture
        = renderer.create_texture(mve::TextureFormat::rgba, mve::Vector3i(sim_props.size), buffer.data());

    global_descriptor.write_binding(frag_shader.descriptor_set(0).binding(1), texture);

    util::FixedLoop fixed_loop(60.0f);

    auto sim_theme = SimDisplayTheme::probability;
    bool cursor_captured = true;

    TextPipeline text_pipeline(renderer, 28);
    float current_offset = 0.0f;
    constexpr float offset = 34.0f;
    TextBuffer fps_text
        = text_pipeline.create_text_buffer(fps_display_text(0), { 0.0f, current_offset }, 1.0f, { 1.0f, 1.0f, 1.0f });
    current_offset += offset;
    TextBuffer tick_text
        = text_pipeline.create_text_buffer(tps_display_text(0), { 0.0f, current_offset }, 1.0f, { 1.0f, 1.0f, 1.0f });
    current_offset += offset;
    TextBuffer lighting_text = text_pipeline.create_text_buffer(
        lighting_display_text(shader_lighting), { 0.0f, current_offset }, 1.0f, { 1.0f, 1.0f, 1.0f });
    current_offset += offset;
    TextBuffer theme_text = text_pipeline.create_text_buffer(
        theme_display_text(sim_theme), { 0.0f, current_offset }, 1.0f, { 1.0f, 1.0f, 1.0f });
    current_offset += offset;
    TextBuffer cursor_text = text_pipeline.create_text_buffer(
        cursor_display_text(cursor_captured), { 0.0f, current_offset }, 1.0f, { 1.0f, 1.0f, 1.0f });
    current_offset += offset;
    TextBuffer fullscreen_text = text_pipeline.create_text_buffer(
        fullscreen_display_text(window.is_fullscreen()), { 0.0f, current_offset }, 1.0f, { 1.0f, 1.0f, 1.0f });

    SimplePipeline simple_pipeline(renderer);

    window.set_resize_callback([&](const mve::Vector2i new_size) {
        renderer.resize(window);
        text_pipeline.resize();
        simple_pipeline.resize(new_size);
        const mve::Matrix4 new_proj = mve::perspective(90.0f, new_size.aspect_ratio(), 0.001f, 100.0f);
        global_ubo.update(vert_shader.descriptor_set(0).binding(0).member("proj").location(), new_proj);
    });
    text_pipeline.resize();
    simple_pipeline.resize(init_screen_size);

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

    auto scenario = Scenario::simple_packet;

    float half_sim_size = static_cast<float>(g_global_data.sim->size()) / 2.0f;
    float third_sim_size = static_cast<float>(g_global_data.sim->size()) / 3.0f;
    mve::Vector3 moving_sphere_center;

    window.disable_cursor();
    int current_frame_count = 0;
    int frame_count;
    auto begin_time = std::chrono::steady_clock::time_point();
    while (!g_global_data.should_exit) {
        if (window.should_close()) {
            g_global_data.should_exit = true;
        }
        window.poll_events();

        if (window.is_key_pressed(mve::Key::p)) {
            g_global_data.paused = !g_global_data.paused;
        }

        if (window.is_key_pressed(mve::Key::f)) {
            if (window.is_fullscreen()) {
                fullscreen_text.update(fullscreen_display_text(false));
                window.windowed();
            }
            else {
                fullscreen_text.update(fullscreen_display_text(true));
                window.fullscreen(true);
            }
        }

        if (window.is_key_pressed(mve::Key::c)) {
            if (cursor_captured) {
                window.enable_cursor();
                cursor_text.update(cursor_display_text(false));
                cursor_captured = false;
            }
            else {
                window.disable_cursor();
                cursor_text.update(cursor_display_text(true));
                cursor_captured = true;
            }
        }

        if (window.is_key_pressed(mve::Key::escape)) {
            g_global_data.should_exit = true;
        }

        if (window.is_key_pressed(mve::Key::one)) {
            g_global_data.sim->clear();
            scenario_simple_packet(*g_global_data.sim);
            scenario = Scenario::simple_packet;
        }
        else if (window.is_key_pressed(mve::Key::two)) {
            g_global_data.sim->clear();
            scenario_simple_packet(*g_global_data.sim);
            scenario_double_slit(*g_global_data.sim);
            scenario = Scenario::double_slit;
        }
        else if (window.is_key_pressed(mve::Key::three)) {
            g_global_data.sim->clear();
            scenario_simple_packet(*g_global_data.sim);
            scenario_double_slit_potential(*g_global_data.sim);
            scenario = Scenario::double_slit_potential;
        }
        else if (window.is_key_pressed(mve::Key::four)) {
            g_global_data.sim->clear();
            scenario_simple_packet(*g_global_data.sim);
            scenario_wall_potential(*g_global_data.sim);
            scenario = Scenario::wall_potential;
        }
        else if (window.is_key_pressed(mve::Key::five)) {
            g_global_data.sim->clear();
            scenario_simple_packet(*g_global_data.sim);
            moving_sphere_center
                = { static_cast<float>(g_global_data.sim->size()) - third_sim_size, half_sim_size, half_sim_size };
            scenario_moving_sphere_potential(*g_global_data.sim, moving_sphere_center);
            scenario = Scenario::moving_sphere_potential;
        }

        if (scenario == Scenario::moving_sphere_potential) {
            constexpr float sphere_speed = 0.1f;
            if (window.is_key_down(mve::Key::numpad_8)) {
                moving_sphere_center.z += sphere_speed;
            }
            else if (window.is_key_down(mve::Key::numpad_5)) {
                moving_sphere_center.z -= sphere_speed;
            }
            if (window.is_key_down(mve::Key::numpad_4)) {
                moving_sphere_center.y += sphere_speed;
            }
            else if (window.is_key_down(mve::Key::numpad_6)) {
                moving_sphere_center.y -= sphere_speed;
            }
            if (window.is_key_down(mve::Key::numpad_7)) {
                moving_sphere_center.x += sphere_speed;
            }
            else if (window.is_key_down(mve::Key::numpad_9)) {
                moving_sphere_center.x -= sphere_speed;
            }
            scenario_moving_sphere_potential(*g_global_data.sim, moving_sphere_center);
        }

        if (window.is_key_pressed(mve::Key::l)) {
            if (shader_lighting) {
                global_ubo.update(shader_lighting_location, 0);
                lighting_text.update(lighting_display_text(false));
                shader_lighting = false;
            }
            else {
                global_ubo.update(shader_lighting_location, 1);
                lighting_text.update(lighting_display_text(true));
                shader_lighting = true;
            }
        }

        if (window.is_key_pressed(mve::Key::t)) {
            if (sim_theme == SimDisplayTheme::probability) {
                theme_text.update(theme_display_text(SimDisplayTheme::components));
                sim_theme = SimDisplayTheme::components;
            }
            else {
                theme_text.update(theme_display_text(SimDisplayTheme::probability));
                sim_theme = SimDisplayTheme::probability;
            }
        }

        if (cursor_captured) {
            camera.update(window);
        }
        fixed_loop.update(1, [&] { camera.fixed_update(window); });

        g_global_data.sim->lock_read();
        fill_buffer_sim(*g_global_data.sim, buffer, sim_theme);
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
        text_pipeline.draw(lighting_text);
        text_pipeline.draw(theme_text);
        text_pipeline.draw(cursor_text);
        text_pipeline.draw(fullscreen_text);

        renderer.end_render_pass();
        renderer.end_frame(window);

        if (std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count() >= 1000000) {
            begin_time = std::chrono::steady_clock::now();
            frame_count = current_frame_count;
            fps_text.update(fps_display_text(frame_count));
            tick_text.update(tps_display_text(g_global_data.sim_frame_count));
            g_global_data.sim_frame_count = 0;
            current_frame_count = 0;
        }
        current_frame_count++;
    }
    g_global_data.sim_thread.join();
}