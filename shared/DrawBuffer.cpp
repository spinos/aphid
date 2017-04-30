#include "DrawBuffer.h"
#ifdef WIN32
#include <gExtension.h>
#else
#include <gl_heads.h>
#endif
#include <BaseTexture.h>

namespace aphid {

DrawBuffer::DrawBuffer() 
{
    m_numIndices = 0;
    m_numVertices = 0;
}

DrawBuffer::~DrawBuffer() 
{
    clearBuffer();
}

void DrawBuffer::clearBuffer()
{
	m_vertices.reset();
	m_normals.reset();
	m_indices.reset();
	m_texcoords.reset();
	m_numVertices = m_numIndices = 0;
}

void DrawBuffer::drawBuffer(const BaseTexture * colorTex) const
{
	if(m_numIndices < 1) return;
    glEnableClientState( GL_VERTEX_ARRAY );
	glVertexPointer( 3, GL_FLOAT, 0, m_vertices.get() );
	
	glEnableClientState( GL_NORMAL_ARRAY );
	glNormalPointer( GL_FLOAT, 0, m_normals.get() );
	
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(2, GL_FLOAT, 0, m_texcoords.get());
	
	if(colorTex) {
		glEnableClientState( GL_COLOR_ARRAY );
		if(colorTex->textureFormat() == BaseTexture::FFloat) 
			glColorPointer(3, GL_FLOAT, 0, colorTex->data());
		else
			glColorPointer(3, GL_UNSIGNED_BYTE, 0, colorTex->data());
	}
	
	glDrawElements( GL_QUADS, m_numIndices, GL_UNSIGNED_INT, m_indices.get());
	
	glDisableClientState( GL_NORMAL_ARRAY );
	glDisableClientState( GL_VERTEX_ARRAY );
	glDisableClientState( GL_TEXTURE_COORD_ARRAY );
	if(colorTex) glDisableClientState( GL_COLOR_ARRAY );
}

void DrawBuffer::createBuffer(unsigned numVertices, unsigned numIndices)
{
    m_vertices.reset(new Vector3F[numVertices]);
	m_normals.reset(new Vector3F[numVertices]);
	m_texcoords.reset(new Float2[numVertices]);
	m_indices.reset(new unsigned[numIndices]);
	m_numVertices = numVertices;
	m_numIndices = numIndices;
}

void DrawBuffer::rebuildBuffer() {}

Vector3F * DrawBuffer::vertices()
{
	return m_vertices.get();
}

Vector3F * DrawBuffer::normals()
{
	return m_normals.get();
}

Float2 * DrawBuffer::texcoords()
{
    return m_texcoords.get();
}

unsigned * DrawBuffer::indices()
{
	return m_indices.get();
}

const unsigned & DrawBuffer::numPoints() const
{
	return m_numVertices;
}

}
