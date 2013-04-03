#include "BaseField.h"
#include "BaseMesh.h"
BaseField::BaseField() : m_color(0), m_value(0) {}
BaseField::~BaseField() 
{
	if(m_color) delete[] m_color;
	if(m_value) delete[] m_value;
}

void BaseField::setMesh(BaseMesh * mesh)
{
	m_mesh = mesh;
	m_numVertices = mesh->getNumVertices();
	m_color = new Vector3F[m_numVertices];
	m_value = new float[m_numVertices];
	for(int i = 0; i < (int)m_numVertices; i++) {
		m_color[i].setZero();
		m_value[i] = 0.f;
	}
}

char BaseField::solve()
{
	return 1;
}

Vector3F * BaseField::getColor() const
{
	return m_color;
}

void BaseField::plotColor()
{
	for(int i = 0; i < (int)m_numVertices; i++) {
		m_color[i].x = m_value[i];
		m_color[i].z = 1.f - m_value[i];
		m_color[i].y = .3f;
	}
}
