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
	MGlobal::displayInfo(MString("master attrib node ")+fmast.name() );
}

void HesperisAttribConnector::ClearMasterNode()
{ MasterAttribNode = MObject::kNullObj; }

void HesperisAttribConnector::Connect(const std::string & name, ANumericAttribute::NumericAttributeType typ,
								MObject & entity, MObject & attr)
{
	if(MasterAttribNode.isNull()) CreateMasterNode();
	
	MString plgName("");
	MString srcName("");
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
	
	if(plgName.length() < 1) return;
	
	MFnDependencyNode fmast(MasterAttribNode);
	MStatus stat;
	MPlug namePlugs = fmast.findPlug(plgName, false, &stat );
	if(!stat) return;
	MPlug outValueArrayPlug = fmast.findPlug(srcName, true, &stat );
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