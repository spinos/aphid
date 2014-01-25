/*
 *  BodyMaps.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/25/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "BodyMaps.h"
#include <PatchTexture.h>
#include <PatchMesh.h>
BodyMaps::BodyMaps() 
{
	PatchTexture * ontag = new PatchTexture;
	ontag->setName("growTag");
	addTexture(ontag);
	PatchTexture * density = new PatchTexture;
	density->setName("growDensity");
	addTexture(density);
	selectTexture(0);
}

BodyMaps::~BodyMaps() {}

void BodyMaps::initializeTextures(PatchMesh * mesh)
{
	for(unsigned i = 0; i < numTextures(); i++) {
		PatchTexture * tex = static_cast<PatchTexture *>(getTexture(i));
		tex->create(mesh);
	}
}

void BodyMaps::fillFaceTagMap(PatchMesh * mesh, char * tag)
{
	PatchTexture * tex = static_cast<PatchTexture *>(getTexture(GrowOnTag));
	const Float3 bright(0.75f, 0.75f, 0.75f);
	const Float3 dark(0.4f, 0.4f, 0.4f);
	
	const unsigned nf = mesh->numQuads();
	for(unsigned i = 0; i < nf; i++) {
		if(tag[i] == 1) tex->fillPatchColor(i, bright);
		else tex->fillPatchColor(i, dark);
	}
}

void BodyMaps::updateFaceTagMap(const std::deque<unsigned> & faces, char * tag)
{
	PatchTexture * tex = static_cast<PatchTexture *>(getTexture(GrowOnTag));
	const Float3 bright(0.75f, 0.75f, 0.75f);
	const Float3 dark(0.4f, 0.4f, 0.4f);
	
	std::deque<unsigned>::const_iterator it = faces.begin();
	for(; it != faces.end(); ++it) {
		if(tag[*it] == 1) tex->fillPatchColor(*it, bright);
		else tex->fillPatchColor(*it, dark);
	}
}
