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
#include <HBase.h>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
BodyMaps::BodyMaps() 
{
	PatchTexture * ontag = new PatchTexture;
	ontag->setName("growTag");
	addTexture(ontag);
	PatchTexture * distribute = new PatchTexture;
	distribute->setName("growDistribute");
	addTexture(distribute);
	PatchTexture * density = new PatchTexture;
	density->setName("growDensity");
	addTexture(density);
	selectTexture(GrowOnTag);
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

bool BodyMaps::isPaintable() const
{
	if(!selectedTexture()) return false;
	if(selectedTexture()->name() == "growTag") return false;
	return true;
}

void BodyMaps::saveTextures(const std::string & groupName)
{
	HBase grpTex(groupName.c_str());
	saveTexture(groupName, GrowDistribute);
	saveTexture(groupName, GrowDensity);
	grpTex.close();
}

void BodyMaps::saveTexture(const std::string & grpName, int texId)
{
	BaseTexture * tex = getTexture(texId);
	if(!tex) return;
	HBase g((boost::format("%1%/%2%") % grpName % tex->name()).str());
	
	int ndata = tex->dataSize();
	if(!g.hasNamedAttr(".size")) g.addIntAttr(".size");
	g.writeIntAttr(".size", &ndata);
	
	if(!g.hasNamedData(".texdata")) g.addCharData(".texdata", ndata);
	g.writeCharData(".texdata", ndata, (char *)tex->data());
	std::cout<<boost::format("write %1%\n") % g.pathToObject();
	g.close();
}

void BodyMaps::loadTextures(const std::string & groupName)
{
	HBase grpTex(groupName.c_str());
	loadTexture(groupName, GrowDistribute);
	loadTexture(groupName, GrowDensity);
	grpTex.close();
}

void BodyMaps::loadTexture(const std::string & grpName, int texId)
{
	BaseTexture * tex = getTexture(texId);
	if(!tex) return;
	HBase g((boost::format("%1%/%2%") % grpName % tex->name()).str());
	
	int ndata = 0;
	if(!g.hasNamedAttr(".size")) {
		std::cout<<"ERROR: tex has no data size.\n";
		g.close();
		return;
	}
	
	g.readIntAttr(".size", &ndata);
	
	if(ndata != tex->dataSize()) {
		std::cout<<"ERROR: tex data size not matched.\n";
		g.close();
		return;
	}
	
	if(!g.hasNamedData(".texdata")) {
		std::cout<<"ERROR: tex has no data.\n";
		g.close();
		return;
	}
	
	g.readCharData(".texdata", ndata, (char *)tex->data());
	tex->setAllWhite(false);
	std::cout<<boost::format("read %1%\n") % g.pathToObject();
	g.close();
}
