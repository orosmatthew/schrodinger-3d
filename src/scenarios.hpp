#pragma once
#include "schrodinger_sim_3d.hpp"

inline void scenario_simple_packet(SchrodingerSim3d& sim)
{
    const int half_size = sim.size() / 2;
    constexpr auto i = std::complex(0.0, 1.0);
    sim.lock_write();
    for (int j = 0; j < sim.size() * sim.size() * sim.size(); ++j) {
        constexpr auto a = 1.0;
        constexpr auto x0 = 10;
        const auto y0 = half_size;
        const auto z0 = half_size;
        constexpr auto sigma_x = 5.0;
        constexpr auto sigma_y = 5.0;
        constexpr auto sigma_z = 5.0;
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

static void scenario_double_slit(SchrodingerSim3d& sim)
{
    const int half_size = sim.size() / 2;
    sim.lock_write();
    const std::array slit_ys = { half_size - 3, half_size - 2, half_size + 2, half_size + 3 };
    for (int y = 0; y < sim.size(); ++y) {
        if (std::ranges::find(slit_ys, y) != slit_ys.end()) {
            continue;
        }
        for (int z = 0; z < sim.size(); ++z) {
            const int wall_x = half_size + 8;
            sim.set_fixed_at({ wall_x, y, z }, true);
        }
    }
    sim.unlock_write();
}

inline void scenario_double_slit_potential(SchrodingerSim3d& sim)
{
    const int half_size = sim.size() / 2;
    sim.lock_write();
    const std::array slit_ys = { half_size - 3, half_size - 2, half_size + 2, half_size + 3 };
    for (int y = 0; y < sim.size(); ++y) {
        if (std::ranges::find(slit_ys, y) != slit_ys.end()) {
            continue;
        }
        for (int z = 0; z < sim.size(); ++z) {
            const int wall_x = half_size + 8;
            sim.set_potential({ wall_x, y, z }, 1.0);
        }
    }
    sim.unlock_write();
}

inline void scenario_wall_potential(SchrodingerSim3d& sim)
{
    const int half_size = sim.size() / 2;
    sim.lock_write();
    for (int y = 0; y < sim.size(); ++y) {
        for (int z = 0; z < sim.size(); ++z) {
            const int wall_x = half_size + 8;
            sim.set_potential({ wall_x, y, z }, 2.0);
        }
    }
    sim.unlock_write();
}

inline void scenario_moving_sphere_potential(SchrodingerSim3d& sim, const mve::Vector3 center)
{
    constexpr double radius = 10.0;
    constexpr double radius_sqrd = radius * radius;
    sim.lock_write();
    for (int x = 0; x < sim.size(); ++x) {
        for (int y = 0; y < sim.size(); ++y) {
            for (int z = 0; z < sim.size(); ++z) {
                const double dx = center.x - static_cast<double>(x);
                const double dy = center.y - static_cast<double>(y);
                const double dz = center.z - static_cast<double>(z);
                if (const double dist_sqrd = dx * dx + dy * dy + dz * dz; dist_sqrd <= radius_sqrd) {
                    constexpr double max_intensity = 2.0;
                    const double intensity = max_intensity * (1.0 - std::sqrt(dist_sqrd) / radius);
                    sim.set_potential({ x, y, z }, intensity);
                }
                else {
                    sim.set_potential({ x, y, z }, 0.0);
                }
            }
        }
    }
    sim.unlock_write();
}