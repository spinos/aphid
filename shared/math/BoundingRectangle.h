/*
 *  BoundingRectangle.h
 *  mallard
 *
 *  Created by jian zhang on 10/3/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_BOUNDING_RECTANGLE_H
#define APH_BOUNDING_RECTANGLE_H
#include <math/Vector2F.h>
#include <string>

namespace aphid {
   
class RectangleI {

	int m_v[4];
	
public:
	void set(int x1, int y1);
	void set(int x0, int y0, int x1, int y1);
	int area() const;
	bool isLandscape() const;
	int width() const;
	int height() const;
	
	void split(RectangleI & r0, RectangleI & r1, float & alpha, bool alongX) const;
	
	friend std::ostream& operator<<(std::ostream &output, const RectangleI & p) {
        output << p.str();
        return output;
    }
	
	const std::string str() const;
};

class BoundingRectangle {
public:
	BoundingRectangle();
	
	void reset();
	void set(float minx, float miny, float maxx, float maxy);
	void update(const Vector2F & p);
	void updateMin(const Vector2F & p);
	void updateMax(const Vector2F & p);
	void translate(const Vector2F & d);
	
	const float getMin(int axis) const;
	const float getMax(int axis) const;
	const float distance(const int &axis) const;
	
	bool isPointInside(const Vector2F & p) const;
	
private:
	float m_data[4];
};

}
#endif