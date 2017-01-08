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
	m_rotOffset.setIdentity();
	m_rotOffset.rotateEuler(rx, ry, rz);
}

void FeatherObject::setRotation(const aphid::Matrix33F & mat)
{
	Matrix44F::setRotation(m_rotOffset * mat);
}

void FeatherObject::getEndPoints(Vector3F * smp) const
{
	smp[0] = m_mesh->points()[0];
	smp[0] = transform(smp[0]);
	smp[1] = m_mesh->points()[m_mesh->numPoints() - 1];
	smp[1] = transform(smp[1]);
}

void FeatherObject::setWarp(Vector3F * dev0)
{
	Matrix44F invRot = *this;
	invRot.inverse();
	dev0[0] = invRot.transformAsNormal(dev0[0]);
	dev0[1] = invRot.transformAsNormal(dev0[1]);
	
	float ang[2];
	ang[0] = calcWarpAngle(dev0[0]);
	ang[1] = calcWarpAngle(dev0[1]);
	
	m_deformer->setWarpAngles(ang);
}

float FeatherObject::calcWarpAngle(Vector3F & vi) const
{
	vi.x *= 0.1f;
	vi.y *= 1.25f;

	vi.x = 0.f;
	
	if(vi.length2() < 1e-5f) {
		return 0.f;
	}
	
	vi.normalize();
	
	float ang = 0.f;
	if(vi.y > 0.f) {
		ang = -acos(vi.dot(Vector3F::ZAxis) );
	}
	
	if(ang > 0.f) {
		ang = 0.f;
	}
	
	return ang;
}
