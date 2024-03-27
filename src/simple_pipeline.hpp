#pragma once

#include "mve/detail/defs.hpp"
#include "mve/renderer.hpp"

class SimplePipeline {
public:
    explicit SimplePipeline(mve::Renderer& renderer);

    void set_view(const mve::Matrix4& mat);

    void resize(mve::Vector2i extent);

    [[nodiscard]] const mve::DescriptorSet& global_descriptor_set() const
    {
        return m_global_descriptor_set;
    }

    static mve::VertexLayout vertex_layout()
    {
        return {
            mve::VertexAttributeType::vec3, // Position
            mve::VertexAttributeType::vec3, // Color
        };
    }

    mve::GraphicsPipeline& pipeline()
    {
        return m_pipeline;
    }

    [[nodiscard]] const mve::GraphicsPipeline& pipeline() const
    {
        return m_pipeline;
    }

    [[nodiscard]] mve::ShaderDescriptorSet model_descriptor_set() const;

    [[nodiscard]] mve::ShaderDescriptorBinding model_uniform_binding() const;

private:
    mve::Renderer* m_renderer;
    mve::Shader m_vert_shader;
    mve::Shader m_frag_shader;
    mve::GraphicsPipeline m_pipeline;
    mve::DescriptorSet m_global_descriptor_set;
    mve::UniformBuffer m_global_ubo;
    mve::UniformLocation m_view_loc;
    mve::UniformLocation m_proj_loc;
};