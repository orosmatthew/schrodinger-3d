#pragma once

#include <complex>
#include <shared_mutex>
#include <vector>

#include <BS_thread_pool.hpp>

#include "common.hpp"

class SchrodingerSim3d {
public:
    struct Properties {
        int size = 512;
        double grid_spacing = 1.0;
        double timestep = 1.0;
        double hbar = 1.0;
        double mass = 1.0;
    };

    explicit SchrodingerSim3d(const Properties& props)
        : c_size(props.size)
        , c_grid_spacing(props.grid_spacing)
        , c_timestep(props.timestep)
        , c_hbar(props.hbar)
        , c_mass(props.mass)
        , m_buffer_present(c_size * c_size * c_size, std::complex(0.0, 0.0))
        , m_buffer_future(c_size * c_size * c_size, std::complex(0.0, 0.0))
        , m_buffer_potential(c_size * c_size * c_size, 0.0)
        , m_buffer_fixed(c_size * c_size * c_size, false)
    {
    }

    [[nodiscard]] size_t pos_to_idx(const mve::Vector3i pos) const
    {
        return pos.z * c_size * c_size + pos.y * c_size + pos.x;
    }

    [[nodiscard]] mve::Vector3i idx_to_pos(const size_t idx) const
    {
        const int z = static_cast<int>(idx / (c_size * c_size));
        const int y = static_cast<int>((idx % (c_size * c_size)) / c_size);
        const int x = static_cast<int>(idx % c_size);
        return { x, y, z };
    }

    [[nodiscard]] bool in_bounds(const mve::Vector3i pos) const
    {
        return pos.x >= 0 && pos.x < c_size && pos.y >= 0 && pos.y < c_size && pos.z >= 0 && pos.z < c_size;
    }

    void update()
    {
        m_buffer_mutex.lock();
        m_thread_pool.detach_blocks<int>(0, c_size * c_size * c_size, [&](const int start, const int end) {
            for (int i = start; i < end; ++i) {
                if (!m_buffer_fixed[i]) {
                    m_buffer_future[i] = future_at_idx(i);
                }
            }
        });
        m_thread_pool.wait();
        m_buffer_mutex.unlock();
        m_buffer_mutex.lock();
        std::swap(m_buffer_present, m_buffer_future);
        m_buffer_mutex.unlock();

        normalize();
    }

    [[nodiscard]] std::complex<double> value_at_idx(const size_t idx) const
    {
        return m_buffer_present[idx];
    }

    [[nodiscard]] std::complex<double> value_at(const mve::Vector3i pos) const
    {
        return value_at_idx(pos_to_idx(pos));
    }

    void set_at(const mve::Vector3i pos, const std::complex<double> value)
    {
        m_buffer_present[pos_to_idx(pos)] = value;
    }

    [[nodiscard]] int size() const
    {
        return c_size;
    }

    [[nodiscard]] double hbar() const
    {
        return c_hbar;
    }

    void set_fixed_at(const mve::Vector3i pos, const bool value)
    {
        m_buffer_fixed[pos_to_idx(pos)] = value;
    }

    [[nodiscard]] bool fixed_at(const mve::Vector3i pos) const
    {
        return m_buffer_fixed[pos_to_idx(pos)];
    }

    [[nodiscard]] bool fixed_at_idx(const size_t idx) const
    {
        return m_buffer_fixed[idx];
    }

    void lock_read()
    {
        m_buffer_mutex.lock_shared();
    }

    void unlock_read()
    {
        m_buffer_mutex.unlock_shared();
    }

    void lock_write()
    {
        m_buffer_mutex.lock();
    }

    void unlock_write()
    {
        m_buffer_mutex.unlock();
    }

    void clear()
    {
        m_buffer_mutex.lock();
        m_buffer_present = std::vector(c_size * c_size * c_size, std::complex(0.0, 0.0));
        m_buffer_future = std::vector(c_size * c_size * c_size, std::complex(0.0, 0.0));
        m_buffer_potential = std::vector(c_size * c_size * c_size, 0.0);
        m_buffer_fixed = std::vector(c_size * c_size * c_size, false);
        m_buffer_mutex.unlock();
    }

private:
    [[nodiscard]] std::complex<double> spatial_derivative_at_idx(const size_t idx) const
    {
        std::array<mve::Vector3i, 6> neighbors {
            { { -1, 0, 0 }, { 1, 0, 0 }, { 0, -1, 0 }, { 0, 1, 0 }, { 0, 0, -1 }, { 0, 0, 1 } }
        };
        auto neighbor_sum = std::complex(0.0, 0.0);
        const auto [x, y, z] = idx_to_pos(idx);
        for (const auto& [n_x, n_y, n_z] : neighbors) {
            if (const mve::Vector3i neighbor { x + n_x, y + n_y, z + n_z }; in_bounds(neighbor)) [[likely]] {
                neighbor_sum += m_buffer_present[pos_to_idx(neighbor)];
            }
        }
        const auto numerator = neighbor_sum - 6.0 * m_buffer_present[idx];
        const auto denominator = c_grid_spacing * c_grid_spacing;
        return numerator / denominator;
    }

    [[nodiscard]] std::complex<double> future_at_idx(const size_t idx) const
    {
        constexpr auto i = std::complex(0.0, 1.0);
        const auto first_term = i * c_timestep * (c_hbar / 2 * c_mass) * spatial_derivative_at_idx(idx);
        const auto second_term = -(i / c_hbar) * c_timestep * m_buffer_potential[idx] * m_buffer_present[idx];
        const auto third_term = m_buffer_present[idx];
        return first_term + second_term + third_term;
    }

    void normalize()
    {
        double sum = 0.0;
        m_buffer_mutex.lock_shared();
        // ReSharper disable once CppTooWideScopeInitStatement
        BS::multi_future<double> block_sums
            = m_thread_pool.submit_blocks<int>(0, c_size * c_size * c_size, [&](const int start, const int end) {
                  double block_sum = 0.0;
                  for (int i = start; i < end; ++i) {
                      block_sum += std::norm(m_buffer_present[i]);
                  }
                  return block_sum;
              });
        for (std::future<double>& future : block_sums) {
            sum += future.get();
        }
        m_buffer_mutex.unlock_shared();
        const double factor = std::sqrt(sum);
        m_buffer_mutex.lock();
        m_thread_pool.detach_blocks(0, c_size * c_size * c_size, [&](const int start, const int end) {
            for (int i = start; i < end; ++i) {
                m_buffer_present[i] /= factor;
            }
        });
        m_thread_pool.wait();
        m_buffer_mutex.unlock();
    }

    const int c_size;
    const double c_grid_spacing;
    const double c_timestep;
    const double c_hbar;
    const double c_mass;
    std::vector<std::complex<double>> m_buffer_present;
    std::vector<std::complex<double>> m_buffer_future;
    std::vector<double> m_buffer_potential;
    std::vector<bool> m_buffer_fixed;
    BS::thread_pool m_thread_pool;
    std::shared_mutex m_buffer_mutex;
};