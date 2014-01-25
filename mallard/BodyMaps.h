/*
 *  BodyMaps.h
 *  mallard
 *
 *  Created by jian zhang on 1/25/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <TextureGroup.h>
#include <TexturePainter.h>
#include <deque>
class PatchMesh;
class BodyMaps : public TextureGroup, public TexturePainter {
public:
	enum MapEntry {
		GrowOnTag = 0,
		GrowDistribute = 1,
		GrowDensity = 2
	};
	
	BodyMaps();
	virtual ~BodyMaps();
	void initializeTextures(PatchMesh * mesh);
protected:
	void fillFaceTagMap(PatchMesh * mesh, char * tag);
	void updateFaceTagMap(const std::deque<unsigned> & faces, char * tag);
private:
};