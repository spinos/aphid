/*
 *  Scene.h
 *  
 *
 *  Created by jian zhang on 6/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef TTG_SCENE_H
#define TTG_SCENE_H
#include <GeoDrawer.h>
namespace ttg {

class Scene {

public:
	Scene();
	virtual ~Scene();
	
	virtual const char * titleStr() const;
	
	virtual bool init();
	virtual bool progressForward();
	virtual bool progressBackward();
	virtual void draw(aphid::GeoDrawer * dr);
	virtual bool viewPerspective() const;
	
};

}
#endif