/*
 *  HLight.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/11/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "HLight.h"
#include <LightGroup.h>
#include <DistantLight.h>
#include <PointLight.h>
#include <SquareLight.h>
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
	writeFloatAttr(".rgb", (float *)(&rgb));
	
	if(!g.hasNamedAttr(".mat")) g.addFloatAttr(".mat", 16);
	Matrix44F mat = l->space();
	writeFloatAttr(".mat", (float *)(&mat));
	
	if(!g.hasNamedAttr(".intensity")) g.addFloatAttr(".intensity");
	float intensity = l->intensity();
	writeFloatAttr(".intensity", &intensity);
	
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
	
}

void HLight::writePointLight(PointLight * l, HBase * g)
{
	
}

void HLight::writeSquareLight(SquareLight * l, HBase * g)
{
	
}

void HLight::readLight(HBase * c, LightGroup * g)
{
	std::cout<<boost::format("read %1%\n") % c->pathToObject();
}
