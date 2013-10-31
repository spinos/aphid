#include "MeshManipulator.h"
#include <BaseMesh.h>
#include <Ray.h>
#include <IntersectionContext.h>
MeshManipulator::MeshManipulator() {}
MeshManipulator::~MeshManipulator() {}

void MeshManipulator::attachTo(BaseMesh * mesh)
{
    m_mesh = mesh;
    m_started = 0;
}

void MeshManipulator::start(const Ray * r)
{
    if(!m_mesh) return;
    IntersectionContext ctx;
    ctx.setComponentFilterType(PrimitiveFilter::TVertex);
    if(!m_mesh->intersect(&ctx)) return;
    
    std::cout<<"sel mesh v "<<ctx.m_componentIdx;
    m_started = 1;
}

void MeshManipulator::perform(const Ray * r)
{
    if(!m_started) return;
}

void MeshManipulator::stop()
{
    m_started = 0;
}
