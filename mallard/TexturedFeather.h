/*
 *  TexturedFeather.h
 *  mallard
 *
 *  Created by jian zhang on 12/24/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "BaseFeather.h"
#include <zEXRImage.h>
class BaseVane;
class TexturedFeather : public BaseFeather {
public:
	TexturedFeather();
	virtual ~TexturedFeather();
	virtual void computeTexcoord();
	virtual void translateUV(const Vector2F & d);
	
	BaseVane * uvVane(short side) const;
	
	static ZEXRImage ColorTextureFile;
	
	void sampleColor(unsigned gridU, unsigned gridV, Vector3F * dst);
protected:

private:
	BaseVane * m_vane;
};