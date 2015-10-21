/*
 *  HesperisAnimIO.cpp
 *  opium
 *
 *  Created by jian zhang on 10/21/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "HesperisAnimIO.h"
#include <AAttributeHelper.h>
#include <HAnimationCurve.h>
#include <maya/MFnAnimCurve.h>
#include <maya/MAnimUtil.h>
#include <boost/format.hpp>

double HesperisAnimIO::SecondsPerFrame = 0.0416667;

bool HesperisAnimIO::WriteAnimation(const MPlug & attrib, const MObject & animCurveObj)
{
	MObject entity = attrib.node();
	const std::string nodeName = H5PathNameTo(entity);
	const std::string shortName(attrib.partialName().asChar());
	std::string animName(MFnDependencyNode(animCurveObj).name().asChar());
	SHelper::validateFilePath(animName);
	const std::string attrName = boost::str(boost::format("%1%|%2%|%3%") % nodeName % shortName % animName );
	
	MStatus status;
	MFnAnimCurve animCurve(animCurveObj, &status);
	if (MS::kSuccess != status) {
		AHelper::Info<std::string >(" not an anim curve ", attrName );
		return false;
	}
	
	MFnAnimCurve::AnimCurveType type = animCurve.animCurveType();

	if(type != MFnAnimCurve::kAnimCurveTU ) {
		AHelper::Info<std::string>(" not a TU anim ", attrName );
		return false;
	}
// only weighted for now
	animCurve.setIsWeighted(true);
	
	AHelper::Info<std::string >(" w anim ", attrName );
	
	AAnimationCurve dataCurve;
	dataCurve.setCurveType(AAnimationCurve::TTU);
		
	const unsigned numKeys = animCurve.numKeyframes();
	for (unsigned i = 0; i < numKeys; i++) {
		AAnimationKey dataKey;
		dataKey._key = (float)animCurve.time(i).value() * SecondsPerFrame; // as seconds
		dataKey._value = animCurve.value(i);
		
		dataKey._inTangentType = TangentTypeAsInt(animCurve.inTangentType(i));
		dataKey._outTangentType = TangentTypeAsInt(animCurve.outTangentType(i));
		
		MAngle angle;
		double weight;
		animCurve.getTangent(i, angle, weight, true);
		
		dataKey._inAngle = angle.as(MAngle::kDegrees);
		dataKey._inWeight = weight;
		
		animCurve.getTangent(i, angle, weight, false);

		dataKey._outAngle = angle.as(MAngle::kDegrees);
		dataKey._outWeight = weight;
		
		dataCurve.addKey(dataKey);
	}
	
	SaveData<HAnimationCurve, AAnimationCurve>(attrName, &dataCurve);
	
	return true;
}

bool HesperisAnimIO::ReadAnimation(HBase * parent, MObject & entity, MObject & attr)
{	
	MPlug panim(entity, attr);
	if(panim.isNull() ) return false;
	
	std::vector<std::string > animNames;
	parent->lsTypedChild<HAnimationCurve>(animNames);
	std::vector<std::string>::const_iterator it = animNames.begin();
	
	for(;it!=animNames.end();++it) {
		AAnimationCurve dataCurve;
		HAnimationCurve grp(*it);
		grp.load(&dataCurve);
		grp.close();

		ProcessAnimationCurve(dataCurve, panim);
	}
	return true;
}

bool HesperisAnimIO::ProcessAnimationCurve(const AAnimationCurve & data, MPlug & dst)
{
	const unsigned n = data.numKeys();
    if(n<1) return false;
	
	RemoveAnimationCurve(dst);
	
	MFnAnimCurve * animCv;
	if( MAnimUtil::isAnimated( dst, false ) ) {
		animCv = new MFnAnimCurve(dst);
	} else {
		animCv = new MFnAnimCurve;
		animCv->create( dst );
	}
	
	animCv->setIsWeighted(true);

	unsigned i = 0;
    for(;i<n;i++) {
        AAnimationKey dataKey = data.key(i);
        
        MTime tmt(dataKey._key / SecondsPerFrame, MTime::uiUnit()); // from seconds
        double valueKey = dataKey._value;
        
        
        MStatus added = animCv->addKeyframe ( tmt, valueKey, NULL);
        if(added != MStatus::kSuccess) {
            MGlobal::displayInfo(MString("cannot add key ") + i);
            return false;
        }
        
// need unlock				
        animCv->setTangentsLocked(i, false);
        animCv->setWeightsLocked(i, false);
        
        MAngle iangle(dataKey._inAngle, MAngle::kDegrees);
        animCv->setTangent( i, iangle, dataKey._inWeight, true);
        MAngle oangle(dataKey._outAngle, MAngle::kDegrees);
        animCv->setTangent( i, oangle, dataKey._outWeight, false);
        
        animCv->setInTangentType(i, IntAsTangentType(dataKey._inTangentType) );
        animCv->setOutTangentType(i, IntAsTangentType(dataKey._outTangentType) );
    }

    delete animCv;

	return true;
}

bool HesperisAnimIO::RemoveAnimationCurve(MPlug & dst)
{
	MObjectArray animation;
	MAnimUtil::findAnimation ( dst, animation );
	if(animation.length() > 0 ) {
		MDGModifier modif;
		modif.deleteNode(animation[0]);
		if(!modif.doIt())
			AHelper::Info<MString>("cannot remove existing animation", dst.name());
	}
	return true;
}

int HesperisAnimIO::TangentTypeAsInt(MFnAnimCurve::TangentType type)
{
	int y = 1;
	switch (type) {
		case MFnAnimCurve::kTangentFixed:
			y = 2;//"tangentFixed";
			break;
		case MFnAnimCurve::kTangentLinear:
			y = 3;//"tangentLinear";
			break;
		case MFnAnimCurve::kTangentFlat:
			y = 4;//"tangentFlat";
			break;
		case MFnAnimCurve::kTangentSmooth:
			y = 5;//"tangentSmooth";
			break;
		case MFnAnimCurve::kTangentStep:
			y = 6;//"tangentStep";
			break;
		case MFnAnimCurve::kTangentStepNext:
			y = 7;//"tangentStepNext";
			break;
		case MFnAnimCurve::kTangentSlow:
			y = 8;//"tangentSlow";
			break;
		case MFnAnimCurve::kTangentFast:
			y = 9;//"tangentFast";
			break;
		case MFnAnimCurve::kTangentClamped:
			y = 10;//"tangentClamped";
			break;
		case MFnAnimCurve::kTangentPlateau:
			y = 11;//"tangentPlateau";
			break;
		default:
			break;
	}
	return y;
}

MFnAnimCurve::TangentType HesperisAnimIO::IntAsTangentType(int x)
{
	MFnAnimCurve::TangentType y = MFnAnimCurve::kTangentGlobal;
	switch (x) {
		case 2 :
			y = MFnAnimCurve::kTangentFixed;
			break;
		case 3 :
			y = MFnAnimCurve::kTangentLinear;
			break;
		case 4 :
			y = MFnAnimCurve::kTangentFlat;
			break;
		case 5 :
			y = MFnAnimCurve::kTangentSmooth;
			break;
		case 6 :
			y = MFnAnimCurve::kTangentStep;
			break;
		case 7 :
			y = MFnAnimCurve::kTangentStepNext;
			break;
		case 8 :
			y = MFnAnimCurve::kTangentSlow;
			break;
		case 9 :
			y = MFnAnimCurve::kTangentFast;
			break;
		case 10:
			y = MFnAnimCurve::kTangentClamped;
			break;
		case 11 :
			y = MFnAnimCurve::kTangentPlateau;
			break;
		default:
			break;
	}
	return y;
}
//:~