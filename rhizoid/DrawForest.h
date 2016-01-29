/*
 *  DrawForest.h
 *  proxyPaint
 *
 *  Created by jian zhang on 1/30/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "Forest.h"

class DrawForest : public sdb::Forest {

public:
    DrawForest();
    virtual ~DrawForest();
    
protected:
    void drawGround();
	
private:
    void drawFaces(Geometry * geo, sdb::Sequence<unsigned> * components);
	
};