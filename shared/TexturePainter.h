/*
 *  TexturePainter.h
 *  aphid
 *
 *  Created by jian zhang on 1/26/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
#include <deque>
#include <ColorBlend.h>
class BaseBrush;
class PatchMesh;
class BaseTexture;
class Patch;
class TexturePainter {
public:
	TexturePainter();
	virtual ~TexturePainter();
	
	void setBrush(BaseBrush * brush);
	void paintOnMeshFaces(PatchMesh * mesh, const std::deque<unsigned> & faceIds, BaseTexture * tex);
	void paintOnFace(const Patch & face, Float3 * tex, const int & ngrid);
protected:

private:
	ColorBlend m_blend;
	BaseBrush * m_brush;
};