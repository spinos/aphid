/*
 *  feather with mesh, transform, rotation offset
 */

#ifndef FEATHER_OBJECT_H
#define FEATHER_OBJECT_H

#include <math/Matrix44F.h>
class FeatherMesh;
class FeatherDeformer;

class FeatherObject : public aphid::Matrix44F {

    FeatherMesh * m_mesh;
    FeatherDeformer * m_deformer;
	aphid::Matrix33F m_rotOffset;
/// for gp predict
	float m_xline;
	
public:
    FeatherObject(FeatherMesh * mesh);
    virtual ~FeatherObject();
    
    const FeatherMesh * mesh() const;
	const FeatherDeformer * deformer() const;
	
	void deform(const aphid::Matrix33F & mat);
	
	void setPredictX(float v);
    const float * predictX() const;
	
	void setRotationOffset(const float & rx,
							const float & ry,
							const float & rz);
							
	void setRotation(const aphid::Matrix33F & mat);
	
/// first and last pnt in world space
	void getEndPoints(aphid::Vector3F * smp) const;
	
/// derivative to previous feather at two points
	void setWarp(aphid::Vector3F * dev0);
	
protected:

private:
	float calcWarpAngle(aphid::Vector3F & vi) const;
	
};

#endif
