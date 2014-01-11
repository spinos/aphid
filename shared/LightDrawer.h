/*
 *  LightDrawer.h
 *  aphid
 *
 *  Created by jian zhang on 1/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "GeoDrawer.h"
#include "AllLight.h"
class LightDrawer : public GeoDrawer {
public:
	LightDrawer();
	virtual ~LightDrawer();
	void drawLights(const LightGroup & grp) const;
	void drawLight(BaseLight * l) const;
	void drawDistantLight(DistantLight * l) const;
	void drawPointLight(PointLight * l) const;
	void drawSquareLight(SquareLight * l) const;
private:

};