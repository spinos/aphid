/*
 *  HAnimationCurve.cpp
 *  aphid
 *
 *  Created by jian zhang on 8/27/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "HAnimationCurve.h"
#include "AAnimationCurve.h"
#include <sstream>
HAnimationCurve::HAnimationCurve(const std::string & path) : HBase(path) 
{}

HAnimationCurve::~HAnimationCurve()
{}
	
char HAnimationCurve::verifyType()
{
	if(!hasNamedAttr(".animcurve_type"))
		return 0;
    return 1;
}

char HAnimationCurve::save(AAnimationCurve * curve)
{
	if(!hasNamedAttr(".animcurve_type"))
        addIntAttr(".animcurve_type");
		
	int t = curve->curveType();
	writeIntAttr(".animcurve_type", &t);
	
	const unsigned n = curve->numKeys();
	unsigned i=0;
	for(;i<n;i++)
		saveKey(i, curve->key(i));
	
	return 1;
}

void HAnimationCurve::saveKey(unsigned i, const AAnimationKey & key)
{
	std::stringstream sst;
	sst<<"key_"<<i;
	std::string keyPath = childPath(sst.str());
	HBase keyGrp(keyPath);
	
	float ftwo[2];
	ftwo[0] = key._key;
	ftwo[1] = key._value;
		
	if(!keyGrp.hasNamedAttr(".key_val"))
        keyGrp.addFloatAttr(".key_val", 2);

	keyGrp.writeFloatAttr(".key_val", ftwo);
	
	if(!keyGrp.hasNamedAttr(".angle"))
        keyGrp.addFloatAttr(".angle", 2);
		
	ftwo[0] = key._inAngle;
	ftwo[1] = key._outAngle;
	
	keyGrp.writeFloatAttr(".angle", ftwo);
	
	if(!keyGrp.hasNamedAttr(".weight"))
        keyGrp.addFloatAttr(".weight", 2);
		
	ftwo[0] = key._inWeight;
	ftwo[1] = key._outWeight;
	keyGrp.writeFloatAttr(".weight", ftwo);
	
	int itwo[2];
	itwo[0] = key._inTangentType;
	itwo[1] = key._outTangentType;
	
	if(!keyGrp.hasNamedAttr(".tangent"))
        keyGrp.addIntAttr(".tangent", 2);
	keyGrp.writeIntAttr(".tangent", itwo);
		
	keyGrp.close();
}

char HAnimationCurve::load(AAnimationCurve * curve)
{
	int t = 0;
	readIntAttr(".animcurve_type", &t);
	if(t == AAnimationCurve::TTA)
		curve->setCurveType(AAnimationCurve::TTA);
	else if(t == AAnimationCurve::TTL)
		curve->setCurveType(AAnimationCurve::TTL);
	else if(t == AAnimationCurve::TTU)
		curve->setCurveType(AAnimationCurve::TTU);
	else 
		std::cout<<"\n warnning: anim curve type is unknown";
		
	const int nc = numChildren();
	int i = 0;
	for(;i<nc;i++) curve->addKey(loadKey(i));
	
	return 1;
}

AAnimationKey HAnimationCurve::loadKey(int i)
{
	std::string keyPath = childPath(i);
	HBase keyGrp(keyPath);
	AAnimationKey key;
	
	int itwo[2];
	keyGrp.readIntAttr(".tangent", itwo);
	
	key._inTangentType = itwo[0];
	key._outTangentType = itwo[1];
	
	float ftwo[2];
	keyGrp.readFloatAttr(".weight", ftwo);
	key._inWeight = ftwo[0];
	key._outWeight = ftwo[1];
	
	keyGrp.readFloatAttr(".angle", ftwo);
	key._inAngle = ftwo[0];
	key._outAngle = ftwo[1];
	
	keyGrp.readFloatAttr(".key_val", ftwo);
	key._key = ftwo[0];
	key._value = ftwo[1];
	
	keyGrp.close();
	return key;
}
//:~