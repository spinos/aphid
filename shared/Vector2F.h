#ifndef VECTOR2F_H
#define VECTOR2F_H

/*
 *  Vector2F.h
 *  easymodel
 *
 *  Created by jian zhang on 10/26/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

class Vector2F{
public:
	Vector2F();
	Vector2F(float vx, float vy);
	Vector2F(const float* p);
	Vector2F(float* p);
	Vector2F(const Vector2F& from, const Vector2F& to);
	
	void set(float vx, float vy);
	
	void operator+=( const Vector2F& other );
	void operator-=( const Vector2F& other );
	//void operator-=( Vector2F other );
	
	void operator/=( const float& scale );	
	void operator*=( const float& scale );
	
	Vector2F operator*( const float& scale ) const;
	Vector2F operator/( const float& scale ) const;
	Vector2F operator*( float& scale ) const;	
	Vector2F operator/( float& scale ) const;
	Vector2F operator*(const Vector2F& other ) const;		
	Vector2F operator+(const Vector2F& other ) const;	
	Vector2F operator-(const Vector2F& other ) const;
	
	float distantTo(const Vector2F & other) const;
	float length() const;
	void reverse();	
	Vector2F reversed() const;
	
	float cross(const Vector2F & b) const;
	void verbose(const char * pref) const;
	float x,y;
};
#endif        //  #ifndef VECTOR2F_H
