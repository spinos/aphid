#include "BaseDeformer.h"

BaseDeformer::BaseDeformer() : m_deformedV(0) {}
BaseDeformer::~BaseDeformer() 
{
	if(m_deformedV) delete[] m_deformedV;
}

void BaseDeformer::setNumVertices(const unsigned & nv)
{
	m_numVertices = nv;
	m_deformedV = new Vector3F[nv];
}

char BaseDeformer::solve()
{
	return 1;
}

Vector3F * BaseDeformer::getDeformedData() const
{
	return m_deformedV;
}
