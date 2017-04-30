#include <ALIMesh.h>
#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreHDF5/All.h>
using namespace Alembic::AbcGeom;

ALIMesh::ALIMesh(Alembic::AbcGeom::IPolyMesh &obj)
{
    m_schema = obj.getSchema();
}

ALIMesh::~ALIMesh() {}

void ALIMesh::verbose()
{
    m_schema.get(m_sample, Abc::ISampleSelector((index_t)0));
    Abc::P3fArraySamplePtr p = m_sample.getPositions();
    std::cout<<"poly mesh\n";
    std::cout<<"p count: "<<p->size()<<std::endl;
    Abc::Int32ArraySamplePtr idx = m_sample.getFaceIndices();
    std::cout<<"index count: "<<idx->size()<<std::endl;
    Abc::Int32ArraySamplePtr fc = m_sample.getFaceCounts();
    std::cout<<"face count: "<<fc->size()<<std::endl;
}
