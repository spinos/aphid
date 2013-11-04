#include "MeshManipulator.h"
#include <BaseMesh.h>
#include <Ray.h>
#include <IntersectionContext.h>
#include <Plane.h>

MeshManipulator::MeshManipulator() 
{
    m_intersect = new IntersectionContext;
}

MeshManipulator::~MeshManipulator() {}

void MeshManipulator::attachTo(BaseMesh * mesh)
{
    m_mesh = mesh;
    m_started = 0;
}

void MeshManipulator::start(const Ray * r)
{
    if(!m_mesh) return;
    m_started = 0;
    
    m_intersect->reset(*r);
    m_intersect->setComponentFilterType(PrimitiveFilter::TVertex);
    
    if(!m_mesh->naiveIntersect(m_intersect)) return;
    
    m_started = 1;
}

void MeshManipulator::perform(const Ray * r)
{
    if(!m_started) return;
    
    Vector3F *p = &m_mesh->vertices()[m_intersect->m_componentIdx];
    
    Plane pl(m_intersect->m_hitN, m_intersect->m_hitP);

    Vector3F hit, d;
    float t;
	if(pl.rayIntersect(*r, hit, t, 1)) {
	    *p += hit - m_intersect->m_hitP;
	    m_intersect->m_hitP = hit;
	}
}

void MeshManipulator::stop()
{
    m_started = 0;
}
