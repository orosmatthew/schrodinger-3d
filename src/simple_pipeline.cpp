#include "simple_pipeline.hpp"
#include "mve/math/math.hpp"

SimplePipeline::SimplePipeline(mve::Renderer& renderer)
    : m_renderer(&renderer)
    , m_vert_shader("../res/bin/shader/simple.vert.spv")
    , m_frag_shader("../res/bin/shader/simple.frag.spv")
    , m_pipeline(renderer, m_vert_shader, m_frag_shader, vertex_layout())
    , m_global_descriptor_set(m_pipeline.create_descriptor_set(m_vert_shader.descriptor_set(0)))
    , m_global_ubo(renderer.create_uniform_buffer(m_vert_shader.descriptor_set(0).binding(0)))
    , m_view_loc(m_vert_shader.descriptor_set(0).binding(0).member("view").location())
    , m_proj_loc(m_vert_shader.descriptor_set(0).binding(0).member("proj").location())
{
    m_global_descriptor_set.write_binding(m_vert_shader.descriptor_set(0).binding(0), m_global_ubo);
}

mve::ShaderDescriptorSet SimplePipeline::model_descriptor_set() const
{
    return m_vert_shader.descriptor_set(1);
}

mve::ShaderDescriptorBinding SimplePipeline::model_uniform_binding() const
{
    return m_vert_shader.descriptor_set(1).binding(0);
}
void SimplePipeline::set_view(const mve::Matrix4& mat)
{
    m_global_ubo.update(m_view_loc, mat);
}
void SimplePipeline::resize(mve::Vector2i extent)
{
    const mve::Matrix4 proj
        = mve::perspective(90.0f, static_cast<float>(extent.x) / static_cast<float>(extent.y), 0.001f, 100.0f);
    m_global_ubo.update(m_proj_loc, proj);
}
