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
class BaseBrush;
class PatchMesh;
class BaseTexture;
class Patch;
class ColorBlend;
class TexturePainter {
public:
    enum PaintMode {
        MReplace = 0,
        MSmooth = 1
    };
	TexturePainter();
	virtual ~TexturePainter();
	
	void setBrush(BaseBrush * brush);
	void paintOnMeshFaces(PatchMesh * mesh, const std::deque<unsigned> & faceIds, BaseTexture * tex);
	void paintOnFace(const Patch & face, Float3 * tex, const int & ngrid);
	
	void setPaintMode(PaintMode m);
	PaintMode paintMode() const;
protected:

private:
    Float3 averageColor(const std::deque<unsigned> & faceIds, BaseTexture * tex) const;
    Float3 m_destinyColor;
    PaintMode m_mode;
	ColorBlend * m_blend;
	BaseBrush * m_brush;
};