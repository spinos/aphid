#pragma once

#include <AllMath.h>

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
	float * texcoords();
	unsigned * indices();
private:
	Vector3F * m_vertices;
    Vector3F * m_normals;
    unsigned * m_indices;
    float * m_texcoords;
    unsigned m_numVertices, m_numIndices;
};
