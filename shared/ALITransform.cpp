#include <ALITransform.h>
#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreHDF5/All.h>
using namespace Alembic::AbcGeom;

ALITransform::ALITransform(Alembic::AbcGeom::IXform &obj) 
{
    m_schema = obj.getSchema();
}

ALITransform::~ALITransform() {}

void ALITransform::verbose()
{
    m_schema.get(m_sample, Abc::ISampleSelector((index_t)0));
    Abc::M44d m = m_sample.getMatrix();
    
    std::cout<<"mat\n";
    std::cout<<m.x[0][0]<<" "<<m.x[0][1]<<" "<<m.x[0][2]<<std::endl;
    std::cout<<m.x[1][0]<<" "<<m.x[1][1]<<" "<<m.x[1][2]<<std::endl;
    std::cout<<m.x[2][0]<<" "<<m.x[2][1]<<" "<<m.x[2][2]<<std::endl;
    std::cout<<m.x[3][0]<<" "<<m.x[3][1]<<" "<<m.x[3][2]<<std::endl;
}
