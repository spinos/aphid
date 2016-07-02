#ifndef LINE_MATH_H
#define LINE_MATH_H

#include <AllMath.h>
namespace aphid {

// http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
inline float distancePointLine(const Vector3F & P0, 
                        const Vector3F & P1, const Vector3F & P2)
{
    return sqrt(((P0 - P1).cross(P0 - P2)).length2() / (P2 - P1).length2()); 
}

inline bool distancePointLineSegment(float & d,
						const Vector3F & P0, 
                        const Vector3F & P1, const Vector3F & P2)
{
	Vector3F v12 = P2 - P1;
	Vector3F v10 = P0 - P1;

	if(v10.dot(v12) <= 0.f ) {
		d = v10.length();
		return false;
	}
		
	Vector3F v20 = P0 - P2;
	
	if(v20.dot(v12) >= 0.f ) {
		d = v20.length();
		return false;
	}
	
    d = sqrt((v10.cross(v20)).length2() / v12.length2() );
	return true;
}

inline void projectPointLineSegment(Vector3F & q,
						const float & d,
						const Vector3F & P0, 
                        const Vector3F & P1, const Vector3F & P2)
{
	Vector3F v10 = P0 - P1;
	float lq1 = sqrt(v10.length2() - d * d);
	Vector3F v12 = P2 - P1;
	q = P1 + v12.normal() * lq1;
}

// http://mathworld.wolfram.com/SkewLines.html
inline bool areIntersectedOrParallelLines(const Vector3F & P1, const Vector3F & P2,
                        const Vector3F & P3, const Vector3F & P4)
{
    return (P3 - P1).dot((P2 - P1).cross(P4 - P3)) == 0.f;
}

inline bool arePerpendicularLines(const Vector3F & P1, const Vector3F & P2,
                        const Vector3F & P3, const Vector3F & P4)
{
    return CloseToZero((P2 - P1).dot(P4 - P3));
}

inline bool areParallelLines(const Vector3F & P1, const Vector3F & P2,
                        const Vector3F & P3, const Vector3F & P4)
{
    return ((P2 - P1).cross(P4 - P3)).length2() < 1e-5;
}

inline bool areIntersectedLines(const Vector3F & P1, const Vector3F & P2,
                        const Vector3F & P3, const Vector3F & P4)
{
    if(areIntersectedOrParallelLines(P1, P2, P3, P4))
        return (! areParallelLines(P1, P2, P3, P4));
    return false;
}

// http://mathworld.wolfram.com/Line-LineDistance.html
inline float distanceBetweenSkewLines(const Vector3F & P1, const Vector3F & P2,
                        const Vector3F & P3, const Vector3F & P4)
{
    const Vector3F n = (P2 - P1).cross(P4 - P3);
    return Absolute((P3 - P1).dot(n)) / n.length();
}

inline float distanceBetweenLines(const Vector3F & P1, const Vector3F & P2,
                        const Vector3F & P3, const Vector3F & P4)
{
    if(areIntersectedOrParallelLines(P1, P2, P3, P4)) {
        if(areParallelLines(P1, P2, P3, P4))
            return distancePointLine(P1, P3, P4);
        return 0.f;
    }
    return distanceBetweenSkewLines(P1, P2, P3, P4);
}

}
#endif        //  #ifndef LINE_MATH_H

