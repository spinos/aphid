#include "Vector3F.h"

#include <cmath>
#include <iostream>

Vector3F Vector3F::XAxis(1.f, 0.f, 0.f);
Vector3F Vector3F::YAxis(0.f, 1.f, 0.f);
Vector3F Vector3F::ZAxis(0.f, 0.f, 1.f);

Vector3F::Vector3F() 
{
	x = y = z = 0.f;
}

Vector3F::Vector3F(const float& vx, const float& vy, const float& vz) 
{
    x = vx;
    y = vy;
    z = vz;
}

Vector3F::Vector3F(const float* p) 
{
    x = p[0]; 
    y = p[1]; 
    z = p[2];
}

Vector3F::Vector3F(float* p) 
{
    x = p[0];
    y = p[1];
    z = p[2];
}

Vector3F::Vector3F(const Vector3F& from, const Vector3F& to) 
{
    x = to.x - from.x; 
    y = to.y - from.y; 
    z = to.z - from.z; 
}

Vector3F::Vector3F(const Vector2F& from)
{
    x = from.x;
    y = from.y;
    z = 0.f;
}

void Vector3F::setZero()
{
	x = y = z = 0.f;
}

void Vector3F::set(float vx, float vy, float vz)
{
    x = vx;
    y = vy;
    z = vz;
}

void Vector3F::setComp(float v, int icomp)
{
	if(icomp < 1) x = v;
	else if(icomp < 2) y = v;
	else z = v;
}

char Vector3F::equals(const Vector3F &other ) const
{
	return (x == other.x && y == other.y && z == other.z);
}

char Vector3F::operator==( const Vector3F& other ) const
{
        return (x == other.x && y == other.y && z == other.z);
}
	
void Vector3F::operator+=( const Vector3F& other )
{
        x += other.x;
        y += other.y;
        z += other.z;
}
	
void Vector3F::operator-=( const Vector3F& other )
{
        x -= other.x;
        y -= other.y;
        z -= other.z;
}
	
void Vector3F::operator/=( const float& scale )
{
    x /= scale;
    y /= scale;
    z /= scale;
}

void Vector3F::operator*=( const float& scale )
{
    x *= scale;
    y *= scale;
    z *= scale;
}

Vector3F Vector3F::operator*(const float& scale ) const
{
        return Vector3F(x*scale,y*scale,z*scale);
}

Vector3F Vector3F::operator*( const Float3 & scale ) const
{
    return Vector3F(x*scale.x,y*scale.y,z*scale.z);
}

Vector3F Vector3F::operator/(const float& scale ) const
{
        return Vector3F(x/scale,y/scale,z/scale);
}

Vector3F Vector3F::operator*( const Vector3F& other ) const
{
        return Vector3F(x*other.x,y* other.y, z* other.z);
}

Vector3F Vector3F::operator+(Vector3F& other ) const
{
        return Vector3F(x+other.x, y+other.y, z+other.z);
}
	
Vector3F Vector3F::operator+(const Vector3F& other ) const
{
        return Vector3F(x+other.x, y+other.y, z+other.z);
}

Vector3F Vector3F::operator-(Vector3F& other ) const
{
        return Vector3F(x-other.x, y-other.y, z-other.z);
}

Vector3F Vector3F::operator-(const Vector3F& other ) const
{
        return Vector3F(x-other.x, y-other.y, z-other.z);
}
	
float Vector3F::length() const
{
        return (float)sqrt(x*x + y*y + z*z);
}
	
float Vector3F::dot(const Vector3F& other) const
{
        return ( x*other.x + y*other.y + z*other.z);
}

Vector3F Vector3F::cross(const Vector3F& other) const
{
        return Vector3F(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
}
	
void Vector3F::normalize()
{
        float len = length();
        if(len > 10e-8)
        {
        x /= len;
        y /= len;
        z /= len;
        }
        else
        {
                x = y = z = 0.577350f;
        }
}

Vector3F Vector3F::normal() const
{
        double mag = sqrt( x * x + y * y + z * z ) + 10e-8;
        return Vector3F(x /(float)mag, y /(float)mag, z /(float)mag);
}
	
void Vector3F::reverse()
{
        x = -x;
        y = -y;
        z = -z;
}
	
Vector3F Vector3F::reversed() const
{
        return Vector3F(-x, -y, -z);
}

void Vector3F::rotateAroundAxis(const Vector3F& axis, float theta)
{
	if(theta==0) return;
	Vector3F ori(x,y,z);
	float l = ori.length();
	ori.normalize();
	
	Vector3F up = axis.cross(ori);
	up.normalize();
	
	Vector3F side = ori - axis*(axis.dot(ori));
	
	up *=side.length();
	
	ori += side*(cos(theta) - 1);
	ori += up*sin(theta);
	
	ori.normalize();
	x = ori.x*l;
	y = ori.y*l;
	z = ori.z*l;
}

Vector3F Vector3F::perpendicular() const
{
	Vector3F ref(0,1,0);
	Vector3F n = normal();
	if(n.y < -0.9f || n.y > 0.9f) ref = Vector3F(1,0,0);
	Vector3F per = cross(ref);
	per.normalize();
	return per;
}

float Vector3F::comp(int dim) const
{
	if(dim < 1) return x;
	if(dim < 2) return y;
	return z;
}

int Vector3F::longestAxis() const
{
	float a = x;
	if(a < 0) a = -a;
	float b = y;
	if(b < 0) b = -b;
	float c = z;
	if(c < 0) c = -c;
	if(a > b && a > c) return 0;
	if(b > c && b > a) return 1;
	return 2;
}

float Vector3F::angleX() const
{
	float r = sqrt(y * y + z * z);
	if(r < 10e-5) return 0.f;
	if(y <= 0.f) return acos(z / r);
	return 6.283f - acos(z / r);
}
	
float Vector3F::angleY() const
{
	float r = sqrt(x * x + z * z);
	if(r < 10e-5) return 0.f;
	if(x > 0.f) return acos(z / r);
	return 6.283f - acos(z / r);
}

float Vector3F::angleBetween(const Vector3F& another, const Vector3F& up) const
{
	const Vector3F nn = another.normal();
	float ang = acos(normal().dot(nn));
	if(up.dot(nn) > 0.f) return ang;
	return -ang;
}

float Vector3F::distance2To(const Vector3F& another) const
{
	return (x - another.x) * (x - another.x) + (y - another.y) * (y - another.y) + (z - another.z) * (z - another.z);
}

void Vector3F::resize(float l)
{
    normalize();
    x *= l;
    y *= l;
    z *= l;
}

void Vector3F::verbose(const char * pref) const
{
	std::cout<<pref<<" ("<<x<<","<<y<<","<<z<<")\n";
}
