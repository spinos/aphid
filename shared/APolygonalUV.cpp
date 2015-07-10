#include "APolygonalUV.h"
#include <BaseBuffer.h>
#include <sstream>

APolygonalUV::APolygonalUV()
{
    m_numCoords = 0;
    m_numIndices = 0;
    m_ucoord = new BaseBuffer;
    m_vcoord = new BaseBuffer;
    m_indices = new BaseBuffer;
}

APolygonalUV::~APolygonalUV()
{
    delete m_ucoord;
    delete m_vcoord;
    delete m_indices;
}

void APolygonalUV::create(unsigned ncoords, unsigned ninds)
{
    m_numCoords = ncoords;
    m_numIndices = ninds;
    m_ucoord->create(ncoords * 4);
    m_vcoord->create(ncoords * 4);
    m_indices->create(ninds * 4);
}

float * APolygonalUV::ucoord() const
{ return (float *)m_ucoord->data(); }

float * APolygonalUV::vcoord() const
{ return (float *)m_vcoord->data(); }

unsigned * APolygonalUV::indices() const
{ return (unsigned *)m_indices->data(); }

const unsigned APolygonalUV::numCoords() const
{ return m_numCoords; }

const unsigned APolygonalUV::numIndices() const
{ return m_numIndices; }

std::string APolygonalUV::verbosestr() const
{
    std::stringstream sst;
    sst<<" poly uv ncoord "<<numCoords()
    <<"\n nfacev "<<numIndices()
    <<"\n";
    return sst.str();
}
//:~
