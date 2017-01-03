#include "FeatherObject.h"
#include "FeatherMesh.h"

FeatherObject::FeatherObject(FeatherMesh * mesh)
{
    m_mesh = mesh;
}

FeatherObject::~FeatherObject()
{
    delete m_mesh;
}

const FeatherMesh * FeatherObject::mesh() const
{ return m_mesh; }
