#include "DrawBuffer.h"
#ifdef WIN32
#include <gExtension.h>
#else
#include <gl_heads.h>
#endif
DrawBuffer::DrawBuffer() 
{
    m_vertices = 0;
    m_normals = 0;
    m_indices = 0;
    m_numIndices = 0;
    m_numVertices = 0;
}

DrawBuffer::~DrawBuffer() 
{
    clearBuffer();
}

void DrawBuffer::clearBuffer()
{
    if(m_vertices) delete[] m_vertices;
	if(m_normals) delete[] m_normals;
	if(m_indices) delete[] m_indices;
	m_numVertices = m_numIndices = 0;
}

void DrawBuffer::drawBuffer() const
{
	if(m_numIndices < 1) return;
    glEnableClientState( GL_VERTEX_ARRAY );
	glVertexPointer( 3, GL_FLOAT, 0, m_vertices );
	
	glEnableClientState( GL_NORMAL_ARRAY );
	glNormalPointer( GL_FLOAT, 0, m_normals );
	
	glDrawElements( GL_QUADS, m_numIndices, GL_UNSIGNED_INT, m_indices);
	
	glDisableClientState( GL_NORMAL_ARRAY );
	glDisableClientState( GL_VERTEX_ARRAY );
}

void DrawBuffer::createBuffer(unsigned numVertices, unsigned numIndices)
{
    clearBuffer();
    m_vertices = new Vector3F[numVertices];
	m_normals = new Vector3F[numVertices];
	m_indices = new unsigned[numIndices];
	m_numVertices = numVertices;
	m_numIndices = numIndices;
}

void DrawBuffer::rebuildBuffer() {}

Vector3F * DrawBuffer::vertices()
{
	return m_vertices;
}

Vector3F * DrawBuffer::normals()
{
	return m_normals;
}

unsigned * DrawBuffer::indices()
{
	return m_indices;
}
