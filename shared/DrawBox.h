/*
 *  DrawBox.h
 *  
 *
 *  Created by jian zhang on 2/4/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
class BoundingBox;

class DrawBox {

public:

	DrawBox();
	virtual ~DrawBox();
	
protected:

	void drawWireBox(const float * center, const float * scale) const;
	void drawSolidBox(const float * center, const float * scale) const;
	void drawBoundingBox(const BoundingBox * box) const;
	
private:
	static const float UnitBoxLine[24][3];
	static const float UnitBoxTriangle[36][3];
	static const float UnitBoxNormal[36][3];
	
};