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
#include <boost/algorithm/string.hpp>
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
	
	if(!g.hasNamedAttr(".typ")) g.addIntAttr(".typ");
	
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
	int typ = 0;
	g->writeIntAttr(".typ", (int *)(&typ));
}

void HLight::writePointLight(PointLight * l, HBase * g)
{
	int typ = 1;
	g->writeIntAttr(".typ", (int *)(&typ));
}

void HLight::writeSquareLight(SquareLight * l, HBase * g)
{
	int typ = 2;
	g->writeIntAttr(".typ", (int *)(&typ));
}

void HLight::readLight(HBase * c, LightGroup * g)
{
	std::cout<<boost::format("read %1%\n") % c->pathToObject();
	if(!c->hasNamedAttr(".typ")) return;
	int typ = 0;
	c->readIntAttr(".typ", &typ);
	BaseLight *l = 0;
	switch (typ) {
		case 0:
			l = readDistantLight(c);
			break;
		case 1:
			l = readPointLight(c);
			break;
		case 2:
			l = readSquareLight(c);
			break;
		default:
			break;
	}
	
	if(!l) return;
	std::string name = c->pathToObject();
	boost::erase_first(name, pathToObject());
	boost::erase_first(name, "/");
	l->setName(name);
	
	Float3 rgb(1.f, 1.f, 1.f);
		
	if(c->hasNamedAttr(".rgb"))
		c->readFloatAttr(".rgb", (float *)(&rgb));
		
	l->setLightColor(rgb);
	
	float intensity = 1.f;
	if(c->hasNamedAttr(".intensity")) 
		c->readFloatAttr(".intensity", &intensity);
	l->setIntensity(intensity);
	
	Vector3F t;
	if(c->hasNamedAttr(".t")) c->readFloatAttr(".t", (float *)(&t));
	
	l->translate(t);
	
	Vector3F rot;
	if(c->hasNamedAttr(".rot")) c->readFloatAttr(".rot", (float *)(&rot));
	
	l->setRotationAngles(rot);
	
	g->addLight(l);
}

BaseLight * HLight::readDistantLight(HBase * c)
{
	DistantLight * l = new DistantLight;
	return l;
}

BaseLight * HLight::readPointLight(HBase * c)
{
	PointLight * l = new PointLight;
	return l;
}

BaseLight * HLight::readSquareLight(HBase * c)
{
	SquareLight * l = new SquareLight;
	return l;
}
