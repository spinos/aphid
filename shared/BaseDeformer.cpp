#include "BaseDeformer.h"

BaseDeformer::BaseDeformer() : m_deformedV(0) {}
BaseDeformer::~BaseDeformer() 
{
	if(m_deformedV) delete[] m_deformedV;
}

void BaseDeformer::setMesh(BaseMesh * mesh)
{
	m_mesh = mesh;
	m_numVertices = mesh->getNumVertices();
	m_deformedV = new Vector3F[m_numVertices];
	reset();
}

char BaseDeformer::solve()
{
	return 1;
}

Vector3F * BaseDeformer::getDeformedData() const
{
	return m_deformedV;
}

void BaseDeformer::reset()
{
	Vector3F *v = m_mesh->getVertices();
	for(int i = 0; i < (int)m_numVertices; i++) {
		m_deformedV[i] = v[i];
	}
}

Vector3F BaseDeformer::restP(unsigned idx) const
{
	return m_mesh->getVertices()[idx];
}