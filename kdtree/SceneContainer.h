/*
 *  SceneContainer.h
 *  qtbullet
 *
 *  Created by jian zhang on 7/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "shapeDrawer.h"
#include <Vector3F.h>
#include <RandomMesh.h>

class SceneContainer {
public:
	SceneContainer();
	virtual ~SceneContainer();
	
	void renderWorld();
	
protected:
	ShapeDrawer* fDrawer;
	RandomMesh * m_mesh;
};