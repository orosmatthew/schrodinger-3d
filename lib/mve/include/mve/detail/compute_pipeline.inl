#pragma once

#include "defs.hpp"

namespace mve {

inline ComputePipeline::ComputePipeline(Renderer& renderer)
{
    *this = std::move(renderer.create_compute_pipeline());
}

inline ComputePipeline& ComputePipeline::operator=(ComputePipeline&& other) noexcept
{
    if (m_renderer != nullptr) {
        m_renderer->destroy(*this);
    }

    m_renderer = other.m_renderer;
    m_handle = other.m_handle;

    other.m_renderer = nullptr;

    return *this;
}
inline bool ComputePipeline::operator==(const ComputePipeline& other) const
{
    return m_renderer == other.m_renderer && m_handle == other.m_handle;
}
inline bool ComputePipeline::operator<(const ComputePipeline& other) const
{
    return m_handle < other.m_handle;
}
inline ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept
    : m_renderer(other.m_renderer)
    , m_handle(other.m_handle)
{
    other.m_renderer = nullptr;
}
inline ComputePipeline::~ComputePipeline()
{
    destroy();
}
inline void ComputePipeline::destroy()
{
    if (m_renderer != nullptr) {
        m_renderer->destroy(*this);
    }
}
inline size_t ComputePipeline::handle() const
{
    return m_handle;
}
inline bool ComputePipeline::is_valid() const
{
    return m_renderer != nullptr;
}

inline DescriptorSet ComputePipeline::create_descriptor_set(const ShaderDescriptorSet& descriptor_set) const
{
    // TODO
    // return m_renderer->create_descriptor_set(*this, descriptor_set);
}
inline void ComputePipeline::invalidate()
{
    m_renderer = nullptr;
}

ComputePipeline::ComputePipeline(Renderer& renderer, const size_t handle)
    : m_renderer(&renderer)
    , m_handle(handle)
{
}

}

template <>
struct std::hash<mve::ComputePipeline> {
    std::size_t operator()(const mve::ComputePipeline& compute_pipeline) const noexcept
    {
        return hash<uint64_t>()(compute_pipeline.handle());
    }
};