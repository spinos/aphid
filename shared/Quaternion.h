#ifndef QUATERNION_H
#define QUATERNION_H

#include <Vector3F.h>

class Quaternion {
public:
    Quaternion();
    Quaternion(float d, float a, float b, float c);
    Quaternion(const float & angle, const Vector3F & axis);
	void set(const float & angle, const Vector3F & axis);
	void set(float d, float a, float b, float c);
	const float magnidute() const;
	void normalize();
	
	Quaternion operator*( const Quaternion & b ) const;	
	Quaternion progress(const Vector3F & angularVelocity, const float & timeStep) const;
	
	float w, x, y, z;
};
#endif        //  #ifndef QUATERNION_H

