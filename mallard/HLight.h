/*
 *  HLight.h
 *  mallard
 *
 *  Created by jian zhang on 1/11/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <HBase.h>
class LightGroup;
class BaseLight;
class DistantLight;
class PointLight;
class SquareLight;
class HLight : public HBase {
public:
	HLight(const std::string & path);
	
	virtual char save(LightGroup * g);
	virtual char load(LightGroup * g);
private:
	void writeLight(BaseLight * l);
	void writeDistantLight(DistantLight * l, HBase * g);
	void writePointLight(PointLight * l, HBase * g);
	void writeSquareLight(SquareLight * l, HBase * g);
	void readLight(HBase * c, LightGroup * g);
};