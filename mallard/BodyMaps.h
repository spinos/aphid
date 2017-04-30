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
	
	void saveTextures(const std::string & groupName);
	void loadTextures(const std::string & groupName);
	
protected:
	void initializeTextures(PatchMesh * mesh);
	void fillFaceTagMap(PatchMesh * mesh, char * tag);
	void updateFaceTagMap(const std::deque<unsigned> & faces, char * tag);
	bool isPaintable() const;
private:
	void saveTexture(const std::string & grpName, int texId);
	void loadTexture(const std::string & grpName, int texId);
};