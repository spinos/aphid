#include "MlFeather.h"

MlFeather::MlFeather() : m_quillz(0){}
MlFeather::~MlFeather() 
{
    if(m_quillz) delete[] m_quillz;
}

void MlFeather::setNumSegment(short x)
{
    m_numSeg = x;
    m_quillz = new float[x];
}
