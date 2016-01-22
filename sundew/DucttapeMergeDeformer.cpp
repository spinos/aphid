/*
 *  DucttapeMergeDeformer.cpp
 *  manuka
 *
 *  Created by jian zhang on 1/22/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "DucttapeMergeDeformer.h"

#include <maya/MItGeometry.h>
#include <AHelper.h>

MTypeId     DucttapeMergeDeformer::id( 0x8e71f55 );

DucttapeMergeDeformer::DucttapeMergeDeformer()
{}

DucttapeMergeDeformer::~DucttapeMergeDeformer()
{}

void* DucttapeMergeDeformer::creator()
{
	return new DucttapeMergeDeformer();
}

MStatus DucttapeMergeDeformer::initialize()
{
	MStatus stat;

	MFnNumericAttribute numericFn;
	MFnTypedAttribute typedAttr;
	
	return MS::kSuccess;
}

MStatus DucttapeMergeDeformer::connectionMade ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	return MPxDeformerNode::connectionMade (plug, otherPlug, asSrc );
}

MStatus DucttapeMergeDeformer::connectionBroken ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	return MPxDeformerNode::connectionBroken ( plug, otherPlug, asSrc );
}

MStatus DucttapeMergeDeformer::deform( MDataBlock& block,
				MItGeometry& iter,
				const MMatrix& m,
				unsigned int multiIndex)
{
	MStatus status;
	MDataHandle envData = block.inputValue(envelope,&status);
	const float env = envData.asFloat();
	if(env < 1e-3f) return status;
	
	for (; !iter.isDone(); iter.next()) {
		//pt = iter.position();
		//iter.setPosition(pt);
	}
	return status;
}