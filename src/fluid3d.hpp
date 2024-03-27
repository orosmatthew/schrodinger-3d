#pragma once

#include <algorithm>
#include <cassert>
#include <vector>

#include <BS_thread_pool.hpp>

class Fluid3D {
public:
    Fluid3D(const int size, const float diffusion, const float viscosity, const int iter)
    {
        assert(size > 0);
        assert(diffusion >= 0.0f);
        assert(viscosity >= 0.0f);
        assert(iter > 0);

        m_size = size;
        m_diff = diffusion;
        m_viscosity = viscosity;

        const std::vector empty(size * size * size, 0.0f);
        m_s = empty;
        m_density = empty;

        m_vel_x = empty;
        m_vel_y = empty;
        m_vel_z = empty;

        m_vel_x_next = empty;
        m_vel_y_next = empty;
        m_vel_z_next = empty;

        m_pressure = empty;
        m_divergence = empty;

        m_tmp = empty;

        m_lin_solve_iterations = iter;
    }

    void step(const float time_step)
    {
        for (size_t i = 0; i < m_size * m_size * m_size; i++) {
            m_density[i] = std::clamp(m_density[i], 0.0f, 10000.0f);
        }

        diffuse(
            BoundaryType::neumann,
            m_thread_pool,
            m_vel_x,
            m_vel_x_next,
            m_tmp,
            m_viscosity,
            time_step,
            m_size,
            m_lin_solve_iterations);
        diffuse(
            BoundaryType::neumann,
            m_thread_pool,
            m_vel_y,
            m_vel_y_next,
            m_tmp,
            m_viscosity,
            time_step,
            m_size,
            m_lin_solve_iterations);
        diffuse(
            BoundaryType::neumann,
            m_thread_pool,
            m_vel_z,
            m_vel_z_next,
            m_tmp,
            m_viscosity,
            time_step,
            m_size,
            m_lin_solve_iterations);

        calc_pressure(
            m_thread_pool,
            m_vel_x_next,
            m_vel_y_next,
            m_vel_z_next,
            m_pressure,
            m_divergence,
            m_tmp,
            m_size,
            m_lin_solve_iterations);
        correct_velocity(m_thread_pool, m_vel_x_next, m_vel_y_next, m_vel_z_next, m_pressure, m_size);

        advect(
            m_thread_pool,
            BoundaryType::neumann,
            m_vel_x_next,
            m_vel_x,
            m_vel_x_next,
            m_vel_y_next,
            m_vel_z_next,
            time_step,
            m_size);
        advect(
            m_thread_pool,
            BoundaryType::neumann,
            m_vel_y_next,
            m_vel_y,
            m_vel_x_next,
            m_vel_y_next,
            m_vel_z_next,
            time_step,
            m_size);
        advect(
            m_thread_pool,
            BoundaryType::neumann,
            m_vel_z_next,
            m_vel_z,
            m_vel_x_next,
            m_vel_y_next,
            m_vel_z_next,
            time_step,
            m_size);

        calc_pressure(
            m_thread_pool, m_vel_x, m_vel_y, m_vel_z, m_pressure, m_divergence, m_tmp, m_size, m_lin_solve_iterations);
        correct_velocity(m_thread_pool, m_vel_x, m_vel_y, m_vel_z, m_pressure, m_size);

        diffuse(
            BoundaryType::fixed,
            m_thread_pool,
            m_density,
            m_s,
            m_tmp,
            m_diff,
            time_step,
            m_size,
            m_lin_solve_iterations);
        advect(m_thread_pool, BoundaryType::fixed, m_s, m_density, m_vel_x, m_vel_y, m_vel_z, time_step, m_size);
        for (size_t i = 0; i < m_size * m_size * m_size; i++) {
            m_density[i] = std::clamp(m_density[i], 0.0f, 10000.0f);
        }
    }

    void add_velocity(
        const int x, const int y, const int z, const float amount_x, const float amount_y, const float amount_z)
    {
        const size_t i = index(x, y, z, m_size);
        m_vel_x[i] += amount_x;
        m_vel_y[i] += amount_y;
        m_vel_z[i] += amount_z;
    }

    void add_density(const int x, const int y, const int z, const float amount)
    {
        m_density[index(x, y, z, m_size)] += amount;
    }

    [[nodiscard]] float density_at(const int x, const int y, const int z) const
    {
        return m_density[index(x, y, z, m_size)];
    }

    [[nodiscard]] float pressure_at(const int x, const int y, const int z) const
    {
        return m_pressure[index(x, y, z, m_size)];
    }

    [[nodiscard]] int size() const
    {
        return m_size;
    }

private:
    struct Vector3 {
        float x;
        float y;
        float z;
    };

    struct Vector3i {
        int x;
        int y;
        int z;
    };

    int m_size;
    float m_diff;
    float m_viscosity;

    BS::thread_pool m_thread_pool {};

    std::vector<float> m_s;
    std::vector<float> m_density;

    std::vector<float> m_vel_x;
    std::vector<float> m_vel_y;
    std::vector<float> m_vel_z;

    std::vector<float> m_vel_x_next;
    std::vector<float> m_vel_y_next;
    std::vector<float> m_vel_z_next;

    std::vector<float> m_pressure;
    std::vector<float> m_divergence;

    std::vector<float> m_tmp;

    int m_lin_solve_iterations;

    static float trilerp(
        const float x,
        const float y,
        const float z,
        const float c000,
        const float c001,
        const float c010,
        const float c011,
        const float c100,
        const float c101,
        const float c110,
        const float c111)
    {
        // Calculate interpolation along x-axis
        const float c00 = c000 * (1 - x) + c100 * x;
        const float c01 = c001 * (1 - x) + c101 * x;
        const float c10 = c010 * (1 - x) + c110 * x;
        const float c11 = c011 * (1 - x) + c111 * x;

        // Calculate interpolation along y-axis
        const float c0 = c00 * (1 - y) + c10 * y;
        const float c1 = c01 * (1 - y) + c11 * y;

        // Calculate interpolation along z-axis
        const float c = c0 * (1 - z) + c1 * z;

        return c;
    };

    static size_t index(const int x, const int y, const int z, const int size)
    {
        return static_cast<size_t>(x) + static_cast<size_t>(y) * size + static_cast<size_t>(z) * size * size;
    }

    static Vector3i index_to_pos(const size_t index, const int size)
    {
        const int z = static_cast<int>(index) / (size * size);
        const int y = static_cast<int>(index) % (size * size) / size;
        const int x = static_cast<int>(index) % (size * size) % size;
        return { x, y, z };
    }

    enum class BoundaryType {
        none, // No boundary condition, used for free surface boundaries
        fixed, // Fixed boundary condition, aka Dirichlet condition
        neumann // Neumann boundary condition, aka zero-gradient condition
    };

    static void set_bnd(const BoundaryType boundary_type, std::vector<float>& x, const int size)
    {
        for (int j = 1; j < size - 1; j++) {
            for (int i = 1; i < size - 1; i++) {
                if (boundary_type == BoundaryType::neumann) {
                    x[index(i, j, 0, size)] = -x[index(i, j, 1, size)];
                    x[index(i, j, size - 1, size)] = -x[index(i, j, size - 2, size)];
                }
                else if (boundary_type == BoundaryType::fixed) {
                    x[index(i, j, 0, size)] = x[index(i, j, 1, size)];
                    x[index(i, j, size - 1, size)] = x[index(i, j, size - 2, size)];
                }
            }
        }

        for (int k = 1; k < size - 1; k++) {
            for (int i = 1; i < size - 1; i++) {
                if (boundary_type == BoundaryType::neumann) {
                    x[index(i, 0, k, size)] = -x[index(i, 1, k, size)];
                    x[index(i, size - 1, k, size)] = -x[index(i, size - 2, k, size)];
                }
                else if (boundary_type == BoundaryType::fixed) {
                    x[index(i, 0, k, size)] = x[index(i, 1, k, size)];
                    x[index(i, size - 1, k, size)] = x[index(i, size - 2, k, size)];
                }
            }
        }

        for (int k = 1; k < size - 1; k++) {
            for (int j = 1; j < size - 1; j++) {
                if (boundary_type == BoundaryType::neumann) {
                    x[index(0, j, k, size)] = -x[index(1, j, k, size)];
                    x[index(size - 1, j, k, size)] = -x[index(size - 2, j, k, size)];
                }
                else if (boundary_type == BoundaryType::fixed) {
                    x[index(0, j, k, size)] = x[index(1, j, k, size)];
                    x[index(size - 1, j, k, size)] = x[index(size - 2, j, k, size)];
                }
            }
        }

        // Boundary condition for corners
        // clang-format off
        if (boundary_type == BoundaryType::fixed) {
            x[index(0, 0, 0, size)]                = 0.33f * (x[index(1, 0, 0, size)]
                                                            + x[index(0, 1, 0, size)]
                                                            + x[index(0, 0, 1, size)]);
            x[index(0, size-1, 0, size)]           = 0.33f * (x[index(1, size-1, 0, size)]
                                                            + x[index(0, size-2, 0, size)]
                                                            + x[index(0, size-1, 1, size)]);
            x[index(0, 0, size-1, size)]           = 0.33f * (x[index(1, 0, size-1, size)]
                                                            + x[index(0, 1, size-1, size)]
                                                            + x[index(0, 0, size, size)]);
            x[index(0, size-1, size-1, size)]      = 0.33f * (x[index(1, size-1, size-1, size)]
                                                            + x[index(0, size-2, size-1, size)]
                                                            + x[index(0, size-1, size-2, size)]);
            x[index(size-1, 0, 0, size)]           = 0.33f * (x[index(size-2, 0, 0, size)]
                                                            + x[index(size-1, 1, 0, size)]
                                                            + x[index(size-1, 0, 1, size)]);
            x[index(size-1, size-1, 0, size)]      = 0.33f * (x[index(size-2, size-1, 0, size)]
                                                            + x[index(size-1, size-2, 0, size)]
                                                            + x[index(size-1, size-1, 1, size)]);
            x[index(size-1, 0, size-1, size)]      = 0.33f * (x[index(size-2, 0, size-1, size)]
                                                            + x[index(size-1, 1, size-1, size)]
                                                            + x[index(size-1, 0, size-2, size)]);
            x[index(size-1, size-1, size-1, size)] = 0.33f * (x[index(size-2, size-1, size-1, size)]
                                                            + x[index(size-1, size-2, size-1, size)]
                                                            + x[index(size-1, size-1, size-2, size)]);
        }
        // clang-format on
        // No boundary condition for corners is needed for Neumann or free surfaces
    }

    // Solve for scalar field for Poisson equation.
    // The Poisson equation is a partial differential equation that relates the second derivative of a scalar
    // function to a source term.
    // L^2*u=f
    // L^2 = Laplacian operator; u = scalar field (dest); f = scalar field (src)
    // The Laplacian operator describes the rate at which a scalar field changes over space (u -> f)
    // This function essentially computes the previous scalar field given the current one.
    // This function only computes the dest value at a single point but will read from adjacent points around.
    // In dest vector:
    //      adj
    //  adj pos adj
    //      adj
    static float linear_solve_point(
        const size_t index,
        const std::vector<float>& dest,
        const std::vector<float>& src,
        const float a,
        const float c_inv,
        const int size)
    {

        const float neighbor_sum = dest[index + 1] + dest[index - 1] + dest[index + size] + dest[index - size]
            + dest[index + (size * size)] + dest[index - (size * size)];

        // Contribution of the laplacian operator to the new value of the current point
        const float laplacian_contribution = a * neighbor_sum;

        return (src[index] + laplacian_contribution) * c_inv;
    }

    static void calc_pressure(
        BS::thread_pool& thread_pool,
        const std::vector<float>& vel_x,
        const std::vector<float>& vel_y,
        const std::vector<float>& vel_z,
        std::vector<float>& pressure,
        std::vector<float>& divergence,
        std::vector<float>& tmp,
        const int size,
        const int iter)
    {
        // Calculate the divergence of the velocity field.
        // Divergence in the velocity field is a scalar value that measures how much the fluid is flowing outward or
        // inward at a given point.
        // (+) away from point
        // (-) toward point
        // (0) not accumulating nor depleting
        // This is used to compute the pressure field which helps to correct for numerical errors by conserving
        // mass, momentum and energy.
        thread_pool
            .parallelize_loop(
                (size - 2) * (size - 2) * (size - 2),
                [&](const int begin, const int end) {
                    for (int i = begin; i < end; i++) {
                        const size_t index = i + (size * size) + size + 1;
                        const float delta_x = vel_x[index + 1] - vel_x[index - 1];
                        const float delta_y = vel_y[index + size] - vel_y[index - size];
                        const float delta_z = vel_z[index + size * size] - vel_y[index - size * size];
                        divergence[index] = (-1.0f / 3.0f) * (delta_x + delta_y + delta_z) / static_cast<float>(size);
                        pressure[index] = 0;
                    }
                })
            .wait();

        set_bnd(BoundaryType::none, divergence, size);
        set_bnd(BoundaryType::none, pressure, size);

        // Scalar constant is used to scale the result of the Poisson equation before updating pressure
        constexpr float scalar_constant = 1.5f;

        // The discretization constant is needed to discretize continuous equations.
        // Represents spacing or step size between points
        constexpr float discretization_constant = 6.0f;

        // Solve for pressure by solving the Poisson equation using the divergence of the velocity field as the source.
        constexpr float c_inv = 1.0f / discretization_constant;
        std::ranges::fill(tmp, 0.0f);
        for (int t = 0; t < iter; t++) {
            thread_pool
                .parallelize_loop(
                    (size - 2) * (size - 2) * (size - 2),
                    [&](const int begin, const int end) {
                        for (int i = begin; i < end; i++) {
                            const int index = i + (size * size) + size + 1;
                            tmp[index] = linear_solve_point(index, pressure, divergence, scalar_constant, c_inv, size);
                        }
                    })
                .wait();
            std::swap(pressure, tmp);
            set_bnd(BoundaryType::none, pressure, size);
        }
    }

    static void correct_velocity(
        BS::thread_pool& thread_pool,
        std::vector<float>& vel_x,
        std::vector<float>& vel_y,
        std::vector<float>& vel_z,
        const std::vector<float>& pressure,
        const int size)
    {
        // Subtract the pressure gradient from the velocity field to ensure incompressibility
        // This ensures that the velocity field remains divergence-free
        thread_pool
            .parallelize_loop(
                (size - 2) * (size - 2) * (size - 2),
                [&](const int begin, const int end) {
                    for (int i = begin; i < end; i++) {
                        const size_t index = i + (size * size) + size + 1;
                        vel_x[index] -= 0.5f * (pressure[index + 1] - pressure[index - 1]) * static_cast<float>(size);
                        vel_y[index]
                            -= 0.5f * (pressure[index + size] - pressure[index - size]) * static_cast<float>(size);
                        vel_z[index] -= 0.5f * (pressure[index + 1] - pressure[index - 1]) * static_cast<float>(size);
                    }
                })
            .wait();

        set_bnd(BoundaryType::neumann, vel_x, size);
        set_bnd(BoundaryType::neumann, vel_y, size);
        set_bnd(BoundaryType::neumann, vel_z, size);
    }

    static void advect(
        BS::thread_pool& thread_pool,
        const BoundaryType boundary_type,
        const std::vector<float>& from,
        std::vector<float>& to,
        const std::vector<float>& vel_x,
        const std::vector<float>& vel_y,
        const std::vector<float>& vel_z,
        const float time_step,
        const int size)
    {
        const Vector3 dt { time_step * (static_cast<float>(size) - 2),
                           time_step * (static_cast<float>(size) - 2),
                           time_step * (static_cast<float>(size) - 2) };

        thread_pool
            .parallelize_loop(
                (size - 2) * (size - 2) * (size - 2),
                [&](const int begin, const int end) {
                    for (int i = begin; i < end; i++) {
                        // Index of current pos
                        const size_t current = i + (size * size) + size + 1;
                        const auto [x, y, z] = index_to_pos(current, size);
                        // displacement = dt * vel
                        const Vector3 displacement {
                            dt.x * vel_x[current], dt.y * vel_y[current], dt.z * vel_z[current]
                        };
                        // new_pos = pos - displacement
                        Vector3 new_pos { static_cast<float>(x) - displacement.x,
                                          static_cast<float>(y) - displacement.y,
                                          static_cast<float>(z) - displacement.z };
                        // Clamp new position to size
                        new_pos.x = std::clamp(new_pos.x, 0.5f, static_cast<float>(size) - 2 + 0.5f);
                        new_pos.y = std::clamp(new_pos.y, 0.5f, static_cast<float>(size) - 2 + 0.5f);
                        new_pos.z = std::clamp(new_pos.z, 0.5f, static_cast<float>(size) - 2 + 0.5f);
                        // new_pos_i = int(floor(new_pos))
                        const Vector3i new_pos_i { static_cast<int>(floorf(new_pos.x)),
                                                   static_cast<int>(floorf(new_pos.y)),
                                                   static_cast<int>(floorf(new_pos.z)) };
                        // offset = new_pos - new_pos_i
                        const Vector3 offset { new_pos.x - static_cast<float>(new_pos_i.x),
                                               new_pos.y - static_cast<float>(new_pos_i.y),
                                               new_pos.z - static_cast<float>(new_pos_i.z) };

                        // Neighboring indices of points in the direction of the displacement/velocity
                        const size_t neighbors[2][2][2]
                            = { { { index(new_pos_i.x, new_pos_i.y, new_pos_i.z, size),
                                    index(new_pos_i.x, new_pos_i.y + 1, new_pos_i.z, size) },
                                  { index(new_pos_i.x + 1, new_pos_i.y, new_pos_i.z, size),
                                    index(new_pos_i.x + 1, new_pos_i.y + 1, new_pos_i.z, size) } },
                                { { index(new_pos_i.x, new_pos_i.y, new_pos_i.z + 1, size),
                                    index(new_pos_i.x, new_pos_i.y + 1, new_pos_i.z + 1, size) },
                                  { index(new_pos_i.x + 1, new_pos_i.y, new_pos_i.z + 1, size),
                                    index(new_pos_i.x + 1, new_pos_i.y + 1, new_pos_i.z + 1, size) } } };

                        // Perform bilinear interpolation between neighbors
                        //                        to[current] = (1.0f - offset.x) * lerp(from[neighbors[0][0]],
                        //                        from[neighbors[0][1]], offset.y)
                        //                            + offset.x * lerp(from[neighbors[1][0]], from[neighbors[1][1]],
                        //                            offset.y);
                        // TODO: Check if the ordering is right here
                        to[current] = trilerp(
                            offset.x,
                            offset.y,
                            offset.z,
                            from[neighbors[0][0][0]],
                            from[neighbors[0][0][1]],
                            from[neighbors[0][1][0]],
                            from[neighbors[0][1][1]],
                            from[neighbors[1][0][0]],
                            from[neighbors[1][0][1]],
                            from[neighbors[1][1][0]],
                            from[neighbors[1][1][1]]);
                    }
                })
            .wait();
        set_bnd(boundary_type, to, size);
    }

    static void diffuse(
        const BoundaryType boundary_type,
        BS::thread_pool& thread_pool,
        const std::vector<float>& from,
        std::vector<float>& to,
        std::vector<float>& tmp,
        const float diffusion_constant,
        const float time_step,
        const int size,
        const int iter)
    {
        // Scaling factor for linear solving.
        // Higher value results in faster rate of diffusion.
        const float a = time_step * diffusion_constant * (static_cast<float>(size) - 2) * (static_cast<float>(size) - 2)
            * (static_cast<float>(size) - 2);
        const float c_inv = 1.0f / (1 + 6 * a);
        std::ranges::fill(tmp, 0.0f);
        for (int t = 0; t < iter; t++) {
            thread_pool
                .parallelize_loop(
                    (size - 2) * (size - 2) * (size - 2),
                    [&](int begin, int end) {
                        for (int current = begin; current < end; current++) {
                            tmp[current + (size * size) + size + 1]
                                = linear_solve_point(current + (size * size) + size + 1, to, from, a, c_inv, size);
                        }
                    })
                .wait();
            std::swap(to, tmp);
            set_bnd(boundary_type, to, size);
        }
    }
};