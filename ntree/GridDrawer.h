/*
 *  GridDrawer.h
 *  testntree
 *
 *  Created by jian zhang on 3/19/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include <DrawBox.h>
#include <VectorArray.h>

namespace aphid {

class GridDrawer : public DrawBox {

public:
	GridDrawer() {}
	virtual ~GridDrawer() {}
	
	template<typename T>
	void drawGrid(T * g);
	
	template<typename T>
	void drawArray(const sdb::VectorArray<T> & arr,
					const Vector3F & origin,
					const float & scaling);
	
};

template<typename T>
void GridDrawer::drawGrid(T * g)
{
	sdb::CellHash * c = g->cells();
	Vector3F l;
    float h;
	c->begin();
	while(!c->end()) {
		l = g->cellCenter(c->key());
		h = g->cellSizeAtLevel(c->value()->level);
        
		drawWireBox((const float *)&l, h);
		
	    c->next();   
	}
}

template<typename T>
void GridDrawer::drawArray(const sdb::VectorArray<T> & arr,
							const Vector3F & origin,
							const float & scaling)
{
	glPushMatrix();
	glTranslatef(origin.x, origin.y, origin.z);
	glScalef(scaling, scaling, scaling);
	const int n = arr.size();
	int i=0;
	for(;i<n;++i) {
		drawBoundingBox(&arr[i]->calculateBBox() );
	}
	glPopMatrix();
}
	
}