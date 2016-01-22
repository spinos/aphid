/*
 *  DucttapeBranchDeformer.cpp
 *  manuka
 *
 *  Created by jian zhang on 1/22/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "DucttapeBranchDeformer.h"
#include <maya/MFnMeshData.h>
#include <maya/MItGeometry.h>
#include <AHelper.h>

MTypeId     DucttapeBranchDeformer::id( 0xd07badb );
MObject DucttapeBranchDeformer::ainmesh;
MObject DucttapeBranchDeformer::aingroupId;

DucttapeBranchDeformer::DucttapeBranchDeformer()
{}

DucttapeBranchDeformer::~DucttapeBranchDeformer()
{}

void* DucttapeBranchDeformer::creator()
{
	return new DucttapeBranchDeformer();
}

MStatus DucttapeBranchDeformer::initialize()
{
	MStatus stat;

	MFnNumericAttribute numericFn;
	MFnTypedAttribute typedAttrFn;
	
	aingroupId = numericFn.create("inGroupId", "ingi", MFnNumericData::kInt, -1);
	numericFn.setStorable(false);
	numericFn.setWritable(true);
	numericFn.setConnectable(true);
	numericFn.setArray(true);
	addAttribute( aingroupId );
	
	ainmesh = typedAttrFn.create("inMesh", "inm", MFnData::kPluginGeometry);
	typedAttrFn.setStorable(false);
	typedAttrFn.setWritable(true);
	typedAttrFn.setConnectable(true);
	typedAttrFn.setArray(true);
	addAttribute( ainmesh );
	
	attributeAffects(ainmesh, outputGeom);
	
	return MS::kSuccess;
}

MStatus DucttapeBranchDeformer::connectionMade ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	return MPxDeformerNode::connectionMade (plug, otherPlug, asSrc );
}

MStatus DucttapeBranchDeformer::connectionBroken ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	return MPxDeformerNode::connectionBroken ( plug, otherPlug, asSrc );
}

MStatus DucttapeBranchDeformer::deform( MDataBlock& block,
				MItGeometry& iter,
				const MMatrix& m,
				unsigned int multiIndex)
{
	MStatus status;
	MDataHandle envData = block.inputValue(envelope,&status);
	const float env = envData.asFloat();
	if(env < 1e-3f) return status;
	
	MArrayDataHandle geomArray = block.inputArrayValue(ainmesh);
	int numSlots = geomArray.elementCount();
	if(numSlots < 1) {
		AHelper::Info<int>("DucttapeBranchDeformer has no input", numSlots);
		return status;
	}
	
	MArrayDataHandle grpIdArray = block.inputArrayValue(aingroupId);
	if(numSlots != grpIdArray.elementCount()) {
		AHelper::Info<int>("DucttapeBranchDeformer has wrong groupId count", grpIdArray.elementCount() );
		return status;
	}
	
	for(int i=0; i < numSlots; i++) {
		int grpId = grpIdArray.inputValue().asInt();
	
		MDataHandle hgeom = geomArray.inputValue();
		MFnGeometryData fgeom(hgeom.data(), &status);
		
		if(fgeom.hasObjectGroup(grpId)) {
			
			MItGeometry inputIter(hgeom, grpId, true, &status);
			if(!status) {
				AHelper::Info<int>("DucttapeBranchDeformer inmesh failed", i);
			}
			
			AHelper::Info<int>("DucttapeBranchDeformer inmesh nv", inputIter.count() );
		}
		else {
			AHelper::Info<int>("DucttapeBranchDeformer input has no group", grpId);
		}
		
		geomArray.next();
		grpIdArray.next();
	}
	
	for (; !iter.isDone(); iter.next()) {
		//pt = iter.position();
		//iter.setPosition(pt);
	}
	return status;
}