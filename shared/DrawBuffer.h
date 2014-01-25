#pragma once

#include <AllMath.h>
#include <boost/scoped_array.hpp>
class DrawBuffer {
public:
    DrawBuffer();
    virtual ~DrawBuffer();
    
    void clearBuffer();
    void drawBuffer() const;
    void createBuffer(unsigned numVertices, unsigned numIndices);
    
    virtual void rebuildBuffer();
    
	Vector3F * vertices();
	Vector3F * normals();
	Float3 * colors();
	Float2 * texcoords();
	unsigned * indices();
	
	const unsigned & numPoints() const;
private:
	boost::scoped_array<Float3> m_colors;
	Vector3F * m_vertices;
    Vector3F * m_normals;
    unsigned * m_indices;
    Float2 * m_texcoords;
    unsigned m_numVertices, m_numIndices;
};
