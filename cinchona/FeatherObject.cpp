#include "FeatherObject.h"
#include "FeatherMesh.h"
#include "FeatherDeformer.h"

using namespace aphid;

FeatherObject::FeatherObject(FeatherMesh * mesh)
{
    m_mesh = mesh;
	m_deformer = new FeatherDeformer(mesh);
}

FeatherObject::~FeatherObject()
{
    delete m_mesh;
	delete m_deformer;
}

const FeatherMesh * FeatherObject::mesh() const
{ return m_mesh; }

const FeatherDeformer * FeatherObject::deformer() const
{ return m_deformer; }

void FeatherObject::deform(const Matrix33F & mat)
{
	m_deformer->deform(mat);
	m_deformer->calculateNormal();
}

void FeatherObject::setPredictX(float v)
{ m_xline = v; }

const float * FeatherObject::predictX() const
{ return &m_xline; }

void FeatherObject::setRotationOffset(const float & rx,
							const float & ry,
							const float & rz)
{
	m_rotOffset.rotateEuler(rx, ry, rz);
}

void FeatherObject::setRotation(const aphid::Matrix33F & mat)
{
	Matrix44F::setRotation(m_rotOffset * mat);
}
