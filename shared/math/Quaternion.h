/*
 *  Quaternion,h
 *
 *  rotation by angle a around the u axis
 *
 */

#ifndef APH_QUATERNION_H
#define APH_QUATERNION_H

#include <math/Vector3F.h>

namespace aphid {

class Quaternion {
public:
    Quaternion();
    Quaternion(float d, float a, float b, float c);
    Quaternion(const float & angle, const Vector3F & axis);
	void set(const float & angle, const Vector3F & axis);
	void set(float d, float a, float b, float c);
	const float magnitude() const;
	void normalize();
	void inverse();
	
	Quaternion operator*( const Quaternion & b ) const;	
	Quaternion progress(const Vector3F & angularVelocity, const float & timeStep) const;
	
	static void Slerp(Quaternion& qOut, 
					const Quaternion& qA,
					const Quaternion& qB,
					const float& t);
					
	float w, x, y, z;
};

}
#endif        //  #ifndef QUATERNION_H

