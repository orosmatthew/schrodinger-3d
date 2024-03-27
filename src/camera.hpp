#pragma once

#include "mve/math/math.hpp"
#include "mve/window.hpp"

class Camera {
public:
    Camera();

    void update(const mve::Window& window);

    void fixed_update(const mve::Window& window);

    [[nodiscard]] mve::Matrix4 view_matrix(const float interpolation_weight) const
    {
        const mve::Matrix4 transform = m_head_transform * m_body_transform;
        const mve::Matrix3 basis = transform.basis();
        const mve::Matrix4 interpolated_transform = mve::Matrix4::from_basis_translation(
            basis, m_prev_pos.linear_interpolate(transform.translation(), interpolation_weight));
        const mve::Matrix4 view = interpolated_transform.inverse().transpose();
        return view;
    }

    [[nodiscard]] mve::Vector3 look_direction() const
    {
        const mve::Matrix4 transform = m_head_transform * m_body_transform;
        const mve::Matrix3 basis = transform.basis().transpose();
        mve::Vector3 direction { 0, 0, -1 };
        direction = direction.rotate(basis);
        return direction.normalize();
    }

private:
    mve::Matrix4 m_body_transform;
    mve::Matrix4 m_head_transform;
    mve::Vector3 m_prev_pos;
    float m_friction;
    float m_acceleration;
    float m_max_speed;
    mve::Vector3 m_velocity;
};