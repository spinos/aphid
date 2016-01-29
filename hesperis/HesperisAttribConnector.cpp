/*
 *  HesperisAttribConnector.cpp
 *  opium
 *
 *  Created by jian zhang on 1/10/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "HesperisAttribConnector.h"
#include <maya/MDGModifier.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MPlug.h>
#include <maya/MGlobal.h>
#include <HObject.h>
#include <AnimUtil.h>
#include <AHelper.h>

MObject HesperisAttribConnector::MasterAttribNode;

void HesperisAttribConnector::CreateMasterNode()
{
	MStatus stat;
    MDGModifier dgModifier;
    MObject omast = dgModifier.createNode("h5AttribCache", &stat);
    dgModifier.doIt();
    
	MFnDependencyNode fmast(omast);
    fmast.findPlug ( MString("cachePath") ).setValue(MString(HObject::FileIO.fileName().c_str()));
    fmast.findPlug ( MString("minFrame") ).setValue(BaseUtil::FirstFrame);
	fmast.findPlug ( MString("maxFrame") ).setValue(BaseUtil::LastFrame);
	
	AnimUtil au;
    au.createSimpleCurve(omast, "currentTime",
                         BaseUtil::FirstFrame, BaseUtil::FirstFrame,
                         BaseUtil::LastFrame, BaseUtil::LastFrame);
    
	MasterAttribNode = omast;
	AHelper::Info<MString>("master attrib node ", fmast.name() );
    AHelper::Info<int>("frame begin ", BaseUtil::FirstFrame );
    AHelper::Info<int>("frame end ", BaseUtil::LastFrame );
}

void HesperisAttribConnector::ClearMasterNode()
{ MasterAttribNode = MObject::kNullObj; }

void HesperisAttribConnector::ConnectNumeric(const std::string & name, ANumericAttribute::NumericAttributeType typ,
								MObject & entity, MObject & attr)
{
	std::string plgName("");
	std::string srcName("");
	switch(typ ) {
		case ANumericAttribute::TByteNumeric:
			plgName = "btnm";
			srcName = "outByte";
			break;
		case ANumericAttribute::TShortNumeric:
			plgName = "stnm";
			srcName = "outShort";
			break;
		case ANumericAttribute::TIntNumeric:
			plgName = "itnm";
			srcName = "outInt";
			break;
		case ANumericAttribute::TBooleanNumeric:
			plgName = "blnm";
			srcName = "outBool";
			break;
		case ANumericAttribute::TFloatNumeric:
			plgName = "ftnm";
			srcName = "outFloat";
			break;
		case ANumericAttribute::TDoubleNumeric:
			plgName = "dbnm";
			srcName = "outDouble";
			break;
		default:
			break;
    }
	
	if(plgName.size() < 1) return;
	
	Connect(name, plgName, srcName, entity, attr);
}

void HesperisAttribConnector::ConnectEnum(const std::string & name, MObject & entity, MObject & attr)
{ Connect(name, "ennm", "outEnum", entity, attr); }

void HesperisAttribConnector::Connect(const std::string & name, const std::string & plgName, const std::string & srcName,
									MObject & entity, MObject & attr)	
{
	if(MasterAttribNode.isNull()) CreateMasterNode();
	
	MFnDependencyNode fmast(MasterAttribNode);
	MStatus stat;
	MPlug namePlugs = fmast.findPlug(plgName.c_str(), false, &stat );
	if(!stat) return;
	MPlug outValueArrayPlug = fmast.findPlug(srcName.c_str(), true, &stat );
	if(!stat) return;
    
	unsigned count = namePlugs.numElements();
	namePlugs.selectAncestorLogicalIndex(count);
    namePlugs.setValue(MString(name.c_str() ) );

    MPlug outValuePlug = outValueArrayPlug.elementByLogicalIndex(count);
    
	MPlug dstPlug(entity, attr);
	
	MDGModifier dgModifier;
    
	dgModifier.connect(outValuePlug, dstPlug);
	dgModifier.doIt();
}
//:~