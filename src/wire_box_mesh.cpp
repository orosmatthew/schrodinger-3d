#include "wire_box_mesh.hpp"
#include "simple_pipeline.hpp"

WireBoxMesh::WireBoxMesh(
    mve::Renderer& renderer,
    mve::GraphicsPipeline& pipeline,
    const mve::ShaderDescriptorSet& set,
    const mve::ShaderDescriptorBinding& uniform_buffer_binding,
    const BoundingBox& box,
    float width,
    mve::Vector3 color)
    : m_renderer(&renderer)
    , m_descriptor_set(pipeline.create_descriptor_set(set))
    , m_uniform_buffer(renderer.create_uniform_buffer(uniform_buffer_binding))
    , m_model_location(uniform_buffer_binding.member("model").location())
{
    m_descriptor_set.write_binding(uniform_buffer_binding, m_uniform_buffer);

    mve::VertexData data(SimplePipeline::vertex_layout());

    Rect3 rect = bounding_box_to_rect3(box);

    std::vector<std::pair<mve::Vector3, mve::Vector3>> edges = rect3_to_edges(rect);

    constexpr float z_offset = 0.005f; // To prevent z-fighting
    mve::Matrix4 scale = mve::Matrix4::from_basis_translation(
        mve::Matrix3::from_scale(mve::Vector3(1.0f - width + z_offset)), { 0, 0, 0 });

    MeshData combined_data;
    for (auto& [fst, snd] : edges) {
        MeshData rect_mesh = create_rect_mesh(fst.transform(scale), snd.transform(scale), width);
        combine_mesh_data(combined_data, rect_mesh);
    }

    for (const mve::Vector3& vertex : combined_data.vertices) {
        data.push_back(vertex);
        data.push_back(color); // TODO: Fix broken color
    }

    m_uniform_buffer.update(m_model_location, mve::Matrix4::identity());

    m_mesh_buffers = { .vertex_buffer = renderer.create_vertex_buffer(data),
                       .index_buffer = renderer.create_index_buffer(combined_data.indices) };
}
void WireBoxMesh::draw(const mve::DescriptorSet& global_set) const
{
    if (m_mesh_buffers.has_value()) {
        m_renderer->bind_descriptor_sets(global_set, m_descriptor_set);
        m_renderer->bind_vertex_buffer(m_mesh_buffers->vertex_buffer);
        m_renderer->draw_index_buffer(m_mesh_buffers->index_buffer);
    }
}
void WireBoxMesh::set_position(const mve::Vector3 position)
{
    m_uniform_buffer.update(m_model_location, mve::Matrix4::identity().translate(position));
}
WireBoxMesh::MeshData WireBoxMesh::create_rect_mesh(const float length, const float width, const mve::Matrix4& matrix)
{
    MeshData mesh_data;
    float hl = length / 2.0f;
    float hw = width / 2.0f;
    // Front
    mesh_data.vertices.emplace_back(-hl, -hw, hw);
    mesh_data.vertices.emplace_back(hl, -hw, hw);
    mesh_data.vertices.emplace_back(hl, -hw, -hw);
    mesh_data.vertices.emplace_back(-hl, -hw, -hw);
    // Back
    mesh_data.vertices.emplace_back(hl, hw, hw);
    mesh_data.vertices.emplace_back(-hl, hw, hw);
    mesh_data.vertices.emplace_back(-hl, hw, -hw);
    mesh_data.vertices.emplace_back(hl, hw, -hw);
    // Left
    mesh_data.vertices.emplace_back(-hl, hw, hw);
    mesh_data.vertices.emplace_back(-hl, -hw, hw);
    mesh_data.vertices.emplace_back(-hl, -hw, -hw);
    mesh_data.vertices.emplace_back(-hl, hw, -hw);
    // Right
    mesh_data.vertices.emplace_back(hl, -hw, hw);
    mesh_data.vertices.emplace_back(hl, hw, hw);
    mesh_data.vertices.emplace_back(hl, hw, -hw);
    mesh_data.vertices.emplace_back(hl, -hw, -hw);
    // Top
    mesh_data.vertices.emplace_back(-hl, hw, hw);
    mesh_data.vertices.emplace_back(hl, hw, hw);
    mesh_data.vertices.emplace_back(hl, -hw, hw);
    mesh_data.vertices.emplace_back(-hl, -hw, hw);
    // Bottom
    mesh_data.vertices.emplace_back(hl, hw, -hw);
    mesh_data.vertices.emplace_back(-hl, hw, -hw);
    mesh_data.vertices.emplace_back(-hl, -hw, -hw);
    mesh_data.vertices.emplace_back(hl, -hw, -hw);

    std::ranges::transform(
        std::as_const(mesh_data.vertices), mesh_data.vertices.begin(), [&](const mve::Vector3 vertex) {
            return vertex.transform(matrix);
        });

    const std::array<uint32_t, 6> quad_indices = { 0, 3, 2, 0, 2, 1 };
    for (int q = 0; q < 6; q++) {
        for (const uint32_t i : quad_indices) {
            mesh_data.indices.push_back(i + (q * 4));
        }
    }
    return mesh_data;
}
WireBoxMesh::MeshData WireBoxMesh::create_rect_mesh(const mve::Vector3 from, const mve::Vector3 to, const float width)
{
    const mve::Vector3 dir = (to - from).normalize().abs();
    const mve::Quaternion quat = mve::Quaternion::from_vector3_to_vector3({ 0, 0, 1 }, dir);
    const mve::Matrix3 basis = mve::Matrix3::from_quaternion(quat);

    const mve::Matrix4 matrix
        = mve::Matrix4::identity().rotate(basis).rotate({ 0, 1, 0 }, mve::radians(90.0f)).translate((to + from) / 2.0f);
    MeshData rect_mesh = create_rect_mesh(from.distance_to(to) + width, width, matrix);
    return rect_mesh;
}

void WireBoxMesh::combine_mesh_data(MeshData& data, const MeshData& other)
{
    const uint32_t indices_offset = data.vertices.size();
    for (const mve::Vector3& vertex : other.vertices) {
        data.vertices.push_back(vertex);
    }
    for (const uint32_t index : other.indices) {
        data.indices.push_back(index + indices_offset);
    }
}
