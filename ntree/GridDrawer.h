/*
 *  GridDrawer.h
 *  testntree
 *
 *  Created by jian zhang on 3/19/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include <DrawBox.h>

namespace aphid {

class GridDrawer : public DrawBox {

public:
	GridDrawer() {}
	virtual ~GridDrawer() {}
	
	template<typename T>
	void drawGrid(T * g);
	
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

}