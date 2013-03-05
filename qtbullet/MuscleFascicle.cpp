#include "MuscleFascicle.h"

MuscleFascicle::MuscleFascicle() {}
MuscleFascicle::~MuscleFascicle() 
{
    m_vertices.clear();
}

void MuscleFascicle::addVertex(const float & x, const float & y, const float & z)
{
    m_vertices.push_back(btVector3(x, y, z));
}

btVector3 MuscleFascicle::vertexAt(int idx) const
{
    return m_vertices.at(idx);
}

int MuscleFascicle::numVertices() const
{
    return m_vertices.size();
}
