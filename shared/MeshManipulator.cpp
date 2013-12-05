#include "MeshManipulator.h"
#include <BaseMesh.h>
#include <Ray.h>
#include <IntersectionContext.h>
#include <Plane.h>
#include <MeshTopology.h>
#include <VertexAdjacency.h>
MeshManipulator::MeshManipulator() 
{
    m_intersect = new IntersectionContext;
	m_topo = new MeshTopology;
	m_started = 0;
	m_mode = 0;
}

MeshManipulator::~MeshManipulator() {}

void MeshManipulator::attachTo(BaseMesh * mesh)
{
    m_mesh = mesh;
	m_topo->buildTopology(m_mesh);
	m_topo->calculateNormal();
}

void MeshManipulator::start(const Ray * r)
{
    if(!m_mesh) return;
    m_started = 0;
    
    m_intersect->reset(*r);
    m_intersect->setComponentFilterType(PrimitiveFilter::TVertex);
    m_intersect->twoSided = 1;
    if(!m_mesh->selectComponent(m_intersect))
		return;
	
    m_started = 1;
}

void MeshManipulator::perform(const Ray * r)
{
    if(!m_started) return;
	if(!m_intersect->m_success) return;

	switch(m_mode) {
		case 1:
			smoothSurface(r);
			break;
		default:
			moveVertex(r);
			break;
	}
}

void MeshManipulator::stop()
{
    m_started = 0;
}

void MeshManipulator::setToMove()
{
	m_mode = 0;
}

void MeshManipulator::setToSmooth()
{
	m_mode = 1;
}

void MeshManipulator::moveVertex(const Ray * r)
{
	Vector3F *p = &m_mesh->vertices()[m_intersect->m_componentIdx];
    
    Plane pl(r->m_dir.reversed(), m_intersect->m_hitP);

    Vector3F hit;
    float t;
	if(pl.rayIntersect(*r, hit, t, 1)) {
	    *p += hit - m_intersect->m_hitP;
	    m_intersect->m_hitP = hit;
	}
}

void MeshManipulator::smoothSurface(const Ray * r)
{
	VertexAdjacency adj = m_topo->getAdjacency(m_intersect->m_componentIdx);
	
	Vector3F *p = &m_mesh->vertices()[m_intersect->m_componentIdx];
    Vector3F d = adj.center() - *p;
	*p += d * .7f;
	
	Plane pl(m_intersect->m_hitN, m_intersect->m_hitP);

    Vector3F hit;
    float t;
	if(!pl.rayIntersect(*r, hit, t, 1)) return;
	
	d = hit - *p;
	float minD = d.length();
	float curD;
	
	VertexAdjacency::VertexNeighbor *neighbor;
    for(neighbor = adj.firstNeighbor(); !adj.isLastNeighbor(); neighbor = adj.nextNeighbor()) {
        d = hit - *(neighbor->v->m_v);
        curD = d.length();
		if(curD < minD) {
			minD = curD;
			m_intersect->m_componentIdx = neighbor->v->getIndex();
		}
    }
}
