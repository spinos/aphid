#ifndef VECTOR3F_H
#define VECTOR3F_H
#include <ATypes.h>
#include <Vector2F.h>
class Vector3F{
public:
	Vector3F();
	Vector3F(const float& vx, const float& vy, const float& vz);
	Vector3F(const float* p);
	Vector3F(float* p);
	Vector3F(const Vector3F& from, const Vector3F& to);
	Vector3F(const Vector2F& from);
	
	void setZero();
	void set(float vx, float vy, float vz);
	void setComp(float v, int icomp);
	
	char equals(const Vector3F &other ) const;
	char operator==( const Vector3F& other ) const;
	char operator!=( const Vector3F& other ) const;
	void operator+=( const Vector3F& other );	
	void operator-=( const Vector3F& other );
	
	void operator/=( const float& scale );	
	void operator*=( const float& scale );
	
	Vector3F operator*( const float& scale ) const;	
	Vector3F operator*( const Float3 & scale ) const;	
	Vector3F operator/( const float& scale ) const;	
	Vector3F operator*( const Vector3F& other ) const;	
	
	Vector3F operator+( Vector3F& other ) const;
	Vector3F operator+( const Vector3F& other ) const;
	
	Vector3F operator-( Vector3F & other ) const;		
	Vector3F operator-( const Vector3F& other ) const;
	
	friend std::ostream& operator<<(std::ostream &output, const Vector3F & p) {
        output << p.str();
        return output;
    }

	const std::string str() const;
	
	const float length() const;
	const float length2() const;
	
	float dot(const Vector3F& other) const;	
	Vector3F cross(const Vector3F& other) const;
	
	void normalize();	
	Vector3F normal() const;
	
	void reverse();	
	Vector3F reversed() const;
	
	void rotateAroundAxis(const Vector3F& axis, float theta);
	Vector3F perpendicular() const;
	
	float comp(int dim) const;
	int longestAxis() const;
	
	float angleX() const;
	float angleY() const;
	
	float angleBetween(const Vector3F& another, const Vector3F& up) const;
	float distanceTo(const Vector3F& another) const;
	float distance2To(const Vector3F& another) const;
	void resize(float l);
	
	void verbose(const char * pref) const;
	
	static Vector3F XAxis;
	static Vector3F YAxis;
	static Vector3F ZAxis;
	static Vector3F Zero;
	static Vector3F One;
	
	float x,y,z;
};
#endif        //  #ifndef VECTOR3F_H

