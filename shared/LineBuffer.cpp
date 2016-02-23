#include "LineBuffer.h"
namespace aphid {

LineBuffer::LineBuffer() 
{
    m_vertices = 0;
    m_numVertices = 0;
}

LineBuffer::~LineBuffer() 
{
    clearBuffer();
}

void LineBuffer::clearBuffer()
{
    if(m_vertices) delete[] m_vertices;
	m_vertices = 0;
	m_numVertices = 0;
}

void LineBuffer::createBuffer(unsigned numVertices)
{
    clearBuffer();
    m_vertices = new Vector3F[numVertices];
	m_numVertices = numVertices;
}

void LineBuffer::rebuildBuffer() {}

Vector3F * LineBuffer::vertices()
{
	return m_vertices;
}

unsigned LineBuffer::numVertices() const
{
    return m_numVertices;
}

}