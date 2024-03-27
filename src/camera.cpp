#include "camera.hpp"
#include "mve/math/math.hpp"

Camera::Camera()
    : m_body_transform(mve::Matrix4::identity().translate(mve::Vector3(1.0f, 0.0f, 0.0f)))
    , m_head_transform(mve::Matrix4::identity())
    , m_prev_pos(mve::Vector3(0, 0, 0))
    , m_friction(0.3f)
    , m_acceleration(0.015f)
    , m_max_speed(0.1f)
{
}
void Camera::update(const mve::Window& window)
{
    const mve::Vector2 mouse_delta = window.mouse_delta();
    m_body_transform = m_body_transform.rotate_local({ 0, 0, 1 }, -mouse_delta.x * 0.001f);
    m_head_transform = m_head_transform.rotate_local({ 1, 0, 0 }, -mouse_delta.y * 0.001f);
    mve::Vector3 head_euler = m_head_transform.euler();
    if (head_euler.x < 0.0f) {
        if (mve::abs(head_euler.x) < mve::abs(head_euler.x + mve::pi)) {
            head_euler.x = 0.0f;
        }
        else {
            head_euler.x = 180.0f;
        }
    }
    if (head_euler.x < mve::radians(0.1f)) {
        m_head_transform = mve::Matrix4::from_basis_translation(
            mve::Matrix3::from_euler({ mve::radians(0.1f), 0, 0 }), m_head_transform.translation());
    }
    if (head_euler.x > mve::radians(179.9f)) {
        m_head_transform = mve::Matrix4::from_basis_translation(
            mve::Matrix3::from_euler({ mve::radians(179.9f), 0, 0 }), m_head_transform.translation());
    }
}
void Camera::fixed_update(const mve::Window& window)
{
    m_prev_pos = m_body_transform.translation();
    mve::Vector3 dir(0.0f);
    if (window.is_key_down(mve::Key::w)) {
        dir.y += 1.0f;
    }
    if (window.is_key_down(mve::Key::s)) {
        dir.y -= 1.0f;
    }
    if (window.is_key_down(mve::Key::a)) {
        dir.x -= 1.0f;
    }
    if (window.is_key_down(mve::Key::d)) {
        dir.x += 1.0f;
    }
    if (window.is_key_down(mve::Key::space)) {
        dir.z += 1.0f;
    }
    if (window.is_key_down(mve::Key::left_shift)) {
        dir.z -= 1.0f;
    }

    dir = dir.rotate(m_body_transform.basis().transpose());
    m_velocity -= (m_velocity * m_friction) * 0.35f;

    m_velocity += dir.normalize() * m_acceleration * mve::clamp((m_max_speed - m_velocity.length()), 0.0f, 1.0f) * 1.5f;

    m_body_transform = m_body_transform.translate(m_velocity);
}
