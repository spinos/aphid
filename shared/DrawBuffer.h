#pragma once

#include <AllMath.h>
#include <boost/scoped_array.hpp>
class BaseTexture;
class DrawBuffer {
public:
    DrawBuffer();
    virtual ~DrawBuffer();
    
    void clearBuffer();
    void drawBuffer(const BaseTexture * colorTex = 0) const;
    void createBuffer(unsigned numVertices, unsigned numIndices);
    
    virtual void rebuildBuffer();
    
	Vector3F * vertices();
	Vector3F * normals();
	Float2 * texcoords();
	unsigned * indices();
	
	const unsigned & numPoints() const;
private:
	boost::scoped_array<Vector3F> m_vertices;
    boost::scoped_array<Vector3F> m_normals;
    boost::scoped_array<unsigned> m_indices;
    boost::scoped_array<Float2> m_texcoords;
    unsigned m_numVertices, m_numIndices;
};
