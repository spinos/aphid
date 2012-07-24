#include <Vector3F.h>

#include <math.h>

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

void Vector3F::set(float vx, float vy, float vz)
{
    x = vx;
    y = vy;
    z = vz;
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

Vector3F Vector3F::operator/(const float& scale ) const
{
        return Vector3F(x/scale,y/scale,z/scale);
}

Vector3F Vector3F::operator*( const Vector3F& other ) const
{
        return Vector3F(x*other.x,y* other.y, z* other.z);
}

Vector3F Vector3F::operator+( Vector3F other )
{
	return Vector3F(x+other.x, y+other.y, z+other.z);
}

Vector3F Vector3F::operator-( Vector3F other )
{
	return Vector3F(x-other.x, y-other.y, z-other.z);
}
	
Vector3F Vector3F::operator+( Vector3F other ) const
{
	return Vector3F(x+other.x, y+other.y, z+other.z);
}
	
Vector3F Vector3F::operator+(Vector3F& other ) const
{
        return Vector3F(x+other.x, y+other.y, z+other.z);
}

Vector3F Vector3F::operator-(Vector3F other ) const
{
        return Vector3F(x-other.x, y-other.y, z-other.z);
}

Vector3F Vector3F::operator-(Vector3F& other ) const
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
