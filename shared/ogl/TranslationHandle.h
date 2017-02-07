/*
 *  TranslationHandle.h
 *  
 *
 *  Created by jian zhang on 1/5/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_OGL_TRANSLATION_HANDLE
#define APH_OGL_TRANSLATION_HANDLE

#include <math/Matrix44F.h>
#include <math/Ray.h>
#include <ogl/DrawArrow.h>
#include <ogl/DrawBox.h>

namespace aphid {

class Plane;

class TranslationHandle : public DrawArrow, public DrawBox {

	Matrix44F * m_space;
	Matrix44F m_invSpace;
	Vector3F m_deltaV;
	Vector3F m_localV;
	float m_radius;
	float m_speed;
	bool m_active;
	
	enum SnapAxis {
		saNone = 0,
		saX = 1,
		saY = 2,
		saZ = 3,
        saXY = 4,
        saYZ = 5,
        saXZ = 6
	};
	
	SnapAxis m_snap;
	
public:
	TranslationHandle(Matrix44F * space);
	virtual ~TranslationHandle();
	
	void setRadius(float x);
	void setSpeed(float x);
	
	bool begin(const Ray * r);
	void end();
	void translate(const Ray * r);
	void draw(const Matrix44F * camspace) const;
	
	void getDeltaTranslation(Vector3F & vec, const float & weight = 1.f) const;

private:
    bool projectLocal(Vector3F & q,
            const Ray * r, const Plane & p1, const Plane & p2);
     bool projectPlaneLocal(Vector3F & q,
            const Ray * r, const Plane & p1);
};

}
#endif
