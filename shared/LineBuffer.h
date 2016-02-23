#pragma once

#include <AllMath.h>
namespace aphid {

class LineBuffer {
public:
    LineBuffer();
    virtual ~LineBuffer();
    
    void clearBuffer();
    void createBuffer(unsigned numVertices);
    
    virtual void rebuildBuffer();
    
	Vector3F * vertices();
	unsigned numVertices() const;

private:
	Vector3F * m_vertices;
    unsigned m_numVertices;
};

}
