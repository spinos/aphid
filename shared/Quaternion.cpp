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

}