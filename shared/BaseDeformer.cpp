#include "BaseDeformer.h"

BaseDeformer::BaseDeformer() : m_deformedV(0), m_restV(0) {}
BaseDeformer::~BaseDeformer() 
{
	if(m_deformedV) delete[] m_deformedV;
	if(m_restV) delete[] m_restV;
}

void BaseDeformer::setMesh(BaseMesh * mesh)
{
	m_mesh = mesh;
	m_numVertices = mesh->getNumVertices();
	m_restV = new Vector3F[m_numVertices];
	m_deformedV = new Vector3F[m_numVertices];
	Vector3F *v = m_mesh->getVertices();
	for(int i = 0; i < (int)m_numVertices; i++) {
		m_restV[i] = v[i];
	}
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
	for(int i = 0; i < (int)m_numVertices; i++) {
		m_deformedV[i] = m_restV[i];
	}
}

Vector3F BaseDeformer::restP(unsigned idx) const
{
	return m_mesh->getVertices()[idx];
}

void BaseDeformer::updateMesh() const
{
	Vector3F *v = m_mesh->vertices();
	for(int i = 0; i < (int)m_numVertices; i++) {
		v[i] = m_deformedV[i];
	}
}
