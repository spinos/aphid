#include "MlFeather.h"

MlFeather::MlFeather() : m_quilly(0), m_vaneVertices(0) {}
MlFeather::~MlFeather() 
{
    if(m_quilly) delete[] m_quilly;
    if(m_vaneVertices) delete[] m_vaneVertices;
}

void MlFeather::createNumSegment(short x)
{
    m_numSeg = x;
    m_quilly = new float[m_numSeg];
    m_vaneVertices = new Vector2F[(m_numSeg + 1) * 6];
}

float * MlFeather::quilly()
{
    return m_quilly;
}

float * MlFeather::getQuilly() const
{
     return m_quilly;
}

Vector2F * MlFeather::vaneAt(short seg, short side)
{
    return &m_vaneVertices[seg * 6 + 3 * side];
}

Vector2F * MlFeather::getVaneAt(short seg, short side) const
{
    return &m_vaneVertices[seg * 6 + 3 * side];
}
