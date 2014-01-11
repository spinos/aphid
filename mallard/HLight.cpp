/*
 *  HLight.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/11/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "HLight.h"
#include <boost/format.hpp>
HLight::HLight(const std::string & path) : HBase(path) {}
	
char HLight::save(LightGroup * g)
{
	int nl = g->numLights();
	
	if(!hasNamedAttr(".nl"))
		addIntAttr(".nl");
		
	writeIntAttr(".nl", &nl);
	std::cout<<" num lights "<<nl<<"\n";
	
	if(nl < 1) return 0;

	for(int i = 0; i < nl; i++)
		writeLight(g->getLight(i));
	
	return 1;
}

char HLight::load(LightGroup * g)
{
	if(!hasNamedAttr(".nl")) {
		std::cout<<"no nl";
		return 0;
	}
	
	int nl = numChildren();
	readIntAttr(".nl", &nl);
	std::cout<<" num lights "<<nl<<"\n";
	
	for(int i = 0; i < nl; i++) {
		HBase c((boost::format("%1%/%2%") % pathToObject() % getChildName(i)).str());
		readLight(&c, g);
		c.close();
	}
	
	return 1;
}

void HLight::writeLight(BaseLight * l)
{
	HBase g((boost::format("%1%/%2%") % pathToObject() % l->name()).str());
	
	if(!g.hasNamedAttr(".rgb")) g.addFloatAttr(".rgb", 3);
	Float3 rgb =l->lightColor();
	g.writeFloatAttr(".rgb", (float *)(&rgb));
	
	if(!g.hasNamedAttr(".intensity")) g.addFloatAttr(".intensity");
	float intensity = l->intensity();
	g.writeFloatAttr(".intensity", &intensity);
	
	if(!g.hasNamedAttr(".t")) g.addFloatAttr(".t", 3);
	Vector3F t = l->translation();
	g.writeFloatAttr(".t", (float *)(&t));
	
	if(!g.hasNamedAttr(".rot")) g.addFloatAttr(".rot", 3);
	Vector3F rot = l->rotationAngles();
	g.writeFloatAttr(".rot", (float *)(&rot));
	
	switch (l->entityType()) {
		case TypedEntity::TDistantLight:
			writeDistantLight(static_cast<DistantLight *>(l), &g);
			break;
		case TypedEntity::TPointLight:
			writePointLight(static_cast<PointLight *>(l), &g);
			break;
		case TypedEntity::TSquareLight:
			writeSquareLight(static_cast<SquareLight *>(l), &g);
			break;
		default:
			break;
	}
	std::cout<<boost::format("write %1%\n") % g.pathToObject();
	g.close();
}

void HLight::writeDistantLight(DistantLight * l, HBase * g)
{
	if(!g->hasNamedAttr(".typ")) g->addIntAttr(".typ");
	int typ = 0;
	g->writeIntAttr(".typ", (int *)(&typ));
}

void HLight::writePointLight(PointLight * l, HBase * g)
{
	if(!g->hasNamedAttr(".typ")) g->addIntAttr(".typ");
	int typ = 1;
	g->writeIntAttr(".typ", (int *)(&typ));
}

void HLight::writeSquareLight(SquareLight * l, HBase * g)
{
	if(!g->hasNamedAttr(".typ")) g->addIntAttr(".typ");
	int typ = 2;
	g->writeIntAttr(".typ", (int *)(&typ));
}

void HLight::readLight(HBase * c, LightGroup * g)
{
	std::cout<<boost::format("read %1%\n") % c->pathToObject();
}
