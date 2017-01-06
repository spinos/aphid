/*
 *  feather with mesh, transform
 */

#ifndef FEATHER_OBJECT_H
#define FEATHER_OBJECT_H

#include <math/Matrix44F.h>
class FeatherMesh;
class FeatherDeformer;

class FeatherObject : public aphid::Matrix44F {

    FeatherMesh * m_mesh;
    FeatherDeformer * m_deformer;
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
	
protected:
};

#endif
