#include "BaseField.h"
#include "BaseMesh.h"
BaseField::BaseField() : m_value(0) {}
BaseField::~BaseField() 
{
	if(m_value) delete[] m_value;
}

void BaseField::setMesh(BaseMesh * mesh)
{
	m_mesh = mesh;
	m_numVertices = mesh->getNumVertices();
	m_value = new Vector3F[m_numVertices];
	for(int i = 0; i < (int)m_numVertices; i++) {
		m_value[i].setZero();
	}
}

char BaseField::solve()
{
	return 1;
}

Vector3F * BaseField::getValue() const
{
	return m_value;
}
