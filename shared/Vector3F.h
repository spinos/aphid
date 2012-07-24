#ifndef VECTOR3F_H
#define VECTOR3F_H

class Vector3F{
public:
	Vector3F();
	Vector3F(const float& vx, const float& vy, const float& vz);
	Vector3F(const float* p);
	Vector3F(float* p);
	Vector3F(const Vector3F& from, const Vector3F& to);
	
	void set(float vx, float vy, float vz);
	
	void operator+=( const Vector3F& other );	
	void operator-=( const Vector3F& other );
	
	void operator/=( const float& scale );	
	void operator*=( const float& scale );
	
	Vector3F operator*( const float& scale ) const;	
	Vector3F operator/( const float& scale ) const;	
	Vector3F operator*( const Vector3F& other ) const;	
	Vector3F operator+( Vector3F other );
	Vector3F operator-( Vector3F other );		
	Vector3F operator+( Vector3F other ) const;	
	Vector3F operator+( Vector3F& other ) const;
	Vector3F operator-( Vector3F other ) const;		
	Vector3F operator-( Vector3F& other ) const;	
	
	float length() const;
	
	float dot(const Vector3F& other) const;	
	Vector3F cross(const Vector3F& other) const;
	
	void normalize();	
	Vector3F normal() const;
	
	void reverse();	
	Vector3F reversed() const;
	
	void rotateAroundAxis(const Vector3F& axis, float theta);
	Vector3F perpendicular() const;

	float x,y,z;
};
#endif        //  #ifndef VECTOR3F_H

