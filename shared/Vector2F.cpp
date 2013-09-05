/*
 *  Vector2F.cpp
 *  easymodel
 *
 *  Created by jian zhang on 10/26/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "Vector2F.h"

#include <cmath>

Vector2F::Vector2F() 
{
	x = y = 0.f;
}

Vector2F::Vector2F(float vx, float vy)
{
	x = vx;
    y = vy;
}

Vector2F::Vector2F(const float* p) 
{
    x = p[0]; 
    y = p[1]; 
}

Vector2F::Vector2F(float* p) 
{
    x = p[0];
    y = p[1];
}

Vector2F::Vector2F(const Vector2F& from, const Vector2F& to) 
{
    x = to.x - from.x; 
    y = to.y - from.y;  
}

void Vector2F::set(float vx, float vy)
{
    x = vx;
    y = vy;
}

void Vector2F::operator+=( const Vector2F& other )
{
        x += other.x;
        y += other.y;
}

void Vector2F::operator-=(Vector2F& other )
{
        x -= other.x;
        y -= other.y;
}

void Vector2F::operator-=( Vector2F other )
{
        x -= other.x;
        y -= other.y;
}

void Vector2F::operator/=( const float& scale )
{
    x /= scale;
    y /= scale;
}

void Vector2F::operator*=( const float& scale )
{
    x /= scale;
    y /= scale;
}

Vector2F Vector2F::operator*(const float& scale ) const
{
        return Vector2F(x*scale,y*scale);
}

Vector2F Vector2F::operator/(const float& scale ) const
{
        return Vector2F(x / scale, y / scale);
}

Vector2F Vector2F::operator*(float& scale ) const
{
        return Vector2F(x*scale,y*scale);
}

Vector2F Vector2F::operator/(float& scale ) const
{
        return Vector2F(x/scale,y/scale);
}
	
Vector2F Vector2F::operator*( Vector2F& other ) const
{
	return Vector2F(x * other.x, y * other.y);
}
	
Vector2F Vector2F::operator+(Vector2F& other ) const
{
        return Vector2F(x+other.x, y+other.y);
}

Vector2F Vector2F::operator-(Vector2F& other ) const
{
        return Vector2F(x-other.x, y-other.y);
}

Vector2F Vector2F::operator+(Vector2F other ) const
{
        return Vector2F(x+other.x, y+other.y);
}

Vector2F Vector2F::operator-(Vector2F other ) const
{
        return Vector2F(x-other.x, y-other.y);
}

Vector2F Vector2F::operator-(Vector2F other )
{
        return Vector2F(x-other.x, y-other.y);
}

float Vector2F::distantTo(const Vector2F & other) const
{
	return sqrt((x - other.x) * (x - other.x) + (y - other.y) * (y - other.y));
}
	
void Vector2F::reverse()
{
        x = -x;
        y = -y;
}
	
Vector2F Vector2F::reversed() const
{
        return Vector2F(-x, -y);
}

float Vector2F::cross(const Vector2F & b) const
{
	return x * b.y - y * b.x;
}
//:~