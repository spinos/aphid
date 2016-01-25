/*
 *  UntangleDeformer.cpp
 *  manuka
 *
 *  Created by jian zhang on 1/22/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "UntangleDeformer.h"
#include <maya/MFnIntArrayData.h>
#include <maya/MItGeometry.h>
#include <AHelper.h>

MTypeId     UntangleDeformer::id( 0x65739df );
MObject UntangleDeformer::atriangleInd;
MObject UntangleDeformer::anumTriangle;

UntangleDeformer::UntangleDeformer()
{}

UntangleDeformer::~UntangleDeformer()
{}

void* UntangleDeformer::creator()
{
	return new UntangleDeformer();
}

MStatus UntangleDeformer::initialize()
{
	MStatus stat;

	MFnNumericAttribute numericFn;
	MFnTypedAttribute typedAttrFn;
	
	MIntArray defaultIntArray;
	MFnIntArrayData intArrayDataFn;
	intArrayDataFn.create( defaultIntArray );
	
	atriangleInd = typedAttrFn.create("inTriangleIndices", "inti", MFnData::kIntArray,
											intArrayDataFn.object() );
	typedAttrFn.setStorable(true);
	addAttribute( atriangleInd );
	
	anumTriangle = numericFn.create("inNumTriangels", "innt", MFnNumericData::kInt);
	numericFn.setDefault(-1);
	numericFn.setStorable(true);
	addAttribute( anumTriangle );
	
	attributeAffects(atriangleInd, outputGeom);
	MGlobal::executeCommand( "makePaintable -attrType multiFloat -sm deformer UntangleDeformer weights" );
	return MS::kSuccess;
}

MStatus UntangleDeformer::connectionMade ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	return MPxDeformerNode::connectionMade (plug, otherPlug, asSrc );
}

MStatus UntangleDeformer::connectionBroken ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	return MPxDeformerNode::connectionBroken ( plug, otherPlug, asSrc );
}

MStatus UntangleDeformer::deform( MDataBlock& block,
				MItGeometry& iter,
				const MMatrix& m,
				unsigned int multiIndex)
{
	MStatus status;
	MDataHandle envData = block.inputValue(envelope,&status);
	const float env = envData.asFloat();
	if(env < 1e-3f) return status;
	
	MPoint pd;
	for (; !iter.isDone(); iter.next()) {
		
		pd = iter.position();
		iter.setPosition(pd);
		
	}
	return status;
}