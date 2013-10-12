/*
 *  MlDrawer.h
 *  mallard
 *
 *  Created by jian zhang on 9/15/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BaseDrawer.h>
#include <BlockDrawBuffer.h>
#include <MlCache.h>

class MlSkin;
class MlCalamus;
class MlTessellate;
class MlFeather;
class BezierDrawer;
class AccPatchMesh;
class MeshTopology;
class BakeDeformer;
class BezierDrawer;
class MlDrawer : public BaseDrawer, public BlockDrawBuffer, public MlCache {
public:
	MlDrawer();
	virtual ~MlDrawer();
	void drawFeather(MlSkin * skin) const;
	void hideAFeather(MlCalamus * c);
	void hideActive(MlSkin * skin);
	void updateActive(MlSkin * skin);
	void updateBuffer(MlCalamus * c);
	void addToBuffer(MlSkin * skin);
	void computeBufferIndirection(MlSkin * skin);
	void rebuildBuffer(MlSkin * skin, bool forced = false);
	void setCurrentFrame(int x);
	void calculateFrame(int x, MlSkin * skin,
	AccPatchMesh * mesh,
	MeshTopology * topo,
	BakeDeformer * deformer,
	BezierDrawer * bezier);
	
protected:
	
private:
    void computeFeather(MlSkin * skin, MlCalamus * c);
	void tessellate(MlFeather * f);
	void writeToCache(MlSkin * skin, const std::string & sliceName);
	void readFromCache(MlSkin * skin, const std::string & sliceName);
private:
	MlTessellate * m_featherTess;
	int m_currentFrame;
	MlSkin * skin;
	AccPatchMesh * mesh;
	MeshTopology * topo;
	BakeDeformer * deformer;
	BezierDrawer * bezier;
};