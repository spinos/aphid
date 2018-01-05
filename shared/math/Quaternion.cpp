#include "Quaternion.h"
#include <cmath>
namespace aphid {

Quaternion::Quaternion()
{ set(1.f, 0.f, 0.f, 0.f) ;}

Quaternion::Quaternion(float d, float a, float b, float c)
{ set(d, a, b, c); }

Quaternion::Quaternion(const float & angle, const Vector3F & axis)
{ set(angle, axis); }

void Quaternion::set(const float & angle, const Vector3F & axis)
{
    const float theta = 0.5f * angle;
    const float s = sin(theta);
    x = axis.x * s;
    y = axis.y * s;
    z = axis.z * s;
    w = cos(theta);
}
	
void Quaternion::set(float d, float a, float b, float c)
{
    w = d;
    x = a;
    y = b;
    z = c;
}

const float Quaternion::magnitude() const
{ return sqrt(w * w + x * x + y * y + z * z); }

void Quaternion::normalize()
{
    const float mag = magnitude();
    x /= mag;
    y /= mag;
    z /= mag;
    w /= mag;
}

/// http://mathworld.wolfram.com/Quaternion.html
Quaternion Quaternion::operator*( const Quaternion & b ) const
{
    const float qw = w * b.w - x * b.x - y * b.y - z * b.z;
	const float qx = w * b.x + x * b.w + y * b.z - z * b.y;
	const float qy = w * b.y - x * b.z + y * b.w + z * b.x;
    const float qz = w * b.z + x * b.y - y * b.x + z * b.w;
    return Quaternion(qw, qx, qy, qz);
}

Quaternion Quaternion::progress(const Vector3F & angularVelocity, const float & timeStep) const
{
	float mva = angularVelocity.length();
	if(mva < 1e-6) return *this;
	Vector3F axis = angularVelocity / mva;
	float theta = mva * timeStep;
	Quaternion q(theta, axis);
	return q * *this;
}

void Quaternion::Slerp(Quaternion& qOut, 
					const Quaternion& qA,
					const Quaternion& qB,
					const float& t)
{
	float		fCosine, fAngle, A, B;

/// parameter checking
	if (t<0.0f || t>1.0f) {
		std::cout<<"\n ERROR Quaternion::Slerp bad parameters";
		qOut.x = 0;
		qOut.y = 0;
		qOut.z = 0;
		qOut.w = 1;
		return;
	}

/// dot product of A and B	
	fCosine = qA.w*qB.w + qA.x*qB.x + qA.y*qB.y + qA.z*qB.z;

	if (fCosine < 0) {
		Quaternion qi;

/// http://www.magic-software.com/Documentation/Quaternions.pdf
/// choose the sign... on q1 so that... the angle
/// between q0 and q1 is acute. This choice avoids extra
/// spinning caused by the interpolated rotations

		qi.x = -qB.x;
		qi.y = -qB.y;
		qi.z = -qB.z;
		qi.w = -qB.w;

		Slerp(qOut, qA, qi, t);
		return;
	}
	
	if(fCosine > 1.f)
		fCosine = 1.f;
		
	fAngle = (float)acos(fCosine);
	
/// A equals B
	if (fAngle==0.0f) {
		qOut = qA;
		return;
	}
	
/// precompute some values
	A = (float)(sin((1.0f-t)*fAngle) / sin(fAngle));
	B = (float)(sin(t*fAngle) / sin(fAngle));

/// compute resulting quaternion
	qOut.x = A * qA.x + B * qB.x;
	qOut.y = A * qA.y + B * qB.y;
	qOut.z = A * qA.z + B * qB.z;
	qOut.w = A * qA.w + B * qB.w;

/// normalise result
	qOut.normalize();
	
}

/// https://cn.mathworks.com/help/aeroblks/quaternioninverse.html
void Quaternion::inverse()
{
	const float tr = w * w + x * x + y * y + z * z;
	w = w / tr;
	x = x / -tr;
	y = y / -tr;
	z = z / -tr;
}

/// https://github.com/millag/DiscreteElasticRods/blob/master/src/ElasticRod.cpp
Quaternion Quaternion::conjugate() const
{
	return Quaternion(w, -x, -y, -z);
}

}