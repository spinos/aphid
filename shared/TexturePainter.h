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
#include <boost/scoped_array.hpp>
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
	char updatePaintPosition();
	void bufferFaces(PatchMesh * mesh, const std::deque<unsigned> & faceIds);
    Float3 averageColor(const std::deque<unsigned> & faceIds, BaseTexture * tex) const;
    boost::scoped_array<Patch> m_faces;
	Float3 m_destinyColor;
	Vector3F m_lastPosition;
	float m_averageFaceSize;
    PaintMode m_mode;
	ColorBlend * m_blend;
	BaseBrush * m_brush;
};