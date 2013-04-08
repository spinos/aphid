#include "BaseField.h"
#include "BaseMesh.h"
BaseField::BaseField() : m_color(0) 
{
	m_activeValue = 0;
}
BaseField::~BaseField() 
{
	if(m_color) delete[] m_color;
	for (std::map<unsigned, float *>::iterator it=m_values.begin(); it!=m_values.end(); ++it)
		delete[] it->second;
}

void BaseField::setMesh(BaseMesh * mesh)
{
	m_mesh = mesh;
	m_numVertices = mesh->getNumVertices();
	m_color = new Vector3F[m_numVertices];
	
	for(int i = 0; i < (int)m_numVertices; i++) {
		m_color[i].setZero();
	}
}

void BaseField::addValue(unsigned idx)
{
	float *v = new float[m_numVertices];
	for(int i = 0; i < (int)m_numVertices; i++) {
		v[i] = 0.f;
	}
	m_values[idx] = v;
}

char BaseField::solve()
{
	return 1;
}

Vector3F * BaseField::getColor() const
{
	return m_color;
}

float * BaseField::value(unsigned idx)
{
	return m_values[idx];
}

float BaseField::getValue(unsigned setIdx, unsigned valIdx)
{
	return value(setIdx)[valIdx];
}

void BaseField::plotColor(unsigned idx)
{
	float *v = m_values[idx];
	for(int i = 0; i < (int)m_numVertices; i++) {
		m_color[i].x = v[i];
		m_color[i].z = 1.f - v[i];
		m_color[i].y = .3f;
	}
}
