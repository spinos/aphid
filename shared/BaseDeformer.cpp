#include "BaseDeformer.h"

BaseDeformer::BaseDeformer() : m_deformedV(0), m_restV(0) 
{
	m_numVertices = 0;
}

BaseDeformer::~BaseDeformer() 
{
	clear();
}

void BaseDeformer::clear()
{
	if(m_restV) delete[] m_restV;
	m_deformedV = m_restV = 0;
	m_numVertices = 0;
}

Vector3F * BaseDeformer::deformedP()
{
	return m_deformedV;
}

Vector3F * BaseDeformer::getDeformedP() const
{
	return m_deformedV;
}

void BaseDeformer::setMesh(BaseMesh * mesh)
{
	clear();
	m_mesh = mesh;
	m_numVertices = mesh->getNumVertices();
	m_deformedV = mesh->vertices();
	m_restV = new Vector3F[m_numVertices];
	Vector3F *v = m_mesh->getVertices();
	for(unsigned i = 0; i < m_numVertices; i++)
		m_restV[i] = v[i];
	
	reset();
}

char BaseDeformer::solve()
{
	return 1;
}

void BaseDeformer::reset()
{
	for(unsigned i = 0; i < m_numVertices; i++)
		m_deformedV[i] = m_restV[i];
}

unsigned BaseDeformer::numVertices() const
{
	return m_numVertices;
}

BoundingBox BaseDeformer::calculateBBox() const
{
    BoundingBox b;
    for(unsigned i = 0; i < m_numVertices; i++) {
        b.updateMin(m_deformedV[i]);
		b.updateMax(m_deformedV[i]);
	}
    return b;
}
