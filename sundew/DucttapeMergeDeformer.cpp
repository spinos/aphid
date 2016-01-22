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
MObject DucttapeMergeDeformer::ainmesh;

DucttapeMergeDeformer::DucttapeMergeDeformer()
{ m_inputGeomIter = NULL; }

DucttapeMergeDeformer::~DucttapeMergeDeformer()
{ if(m_inputGeomIter) delete m_inputGeomIter; }

void* DucttapeMergeDeformer::creator()
{
	return new DucttapeMergeDeformer();
}

MStatus DucttapeMergeDeformer::initialize()
{
	MStatus stat;

	MFnNumericAttribute numericFn;
	MFnTypedAttribute typedAttrFn;
	
	ainmesh = typedAttrFn.create("inMesh", "inm", MFnData::kMesh);
	typedAttrFn.setStorable(false);
	typedAttrFn.setWritable(true);
	typedAttrFn.setConnectable(true);
	addAttribute( ainmesh );
	
	attributeAffects(ainmesh, outputGeom);
	MGlobal::executeCommand( "makePaintable -attrType multiFloat -sm deformer ducttapeMergeDeformer weights" );
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
	
	if(multiIndex == 0) {
		if(m_inputGeomIter) {
			delete m_inputGeomIter;
			m_inputGeomIter = NULL;
		}
		
		MDataHandle hmesh = block.inputValue(ainmesh);
		
		MItGeometry * aiter = new MItGeometry( hmesh, true, &status );
		if(!status) {
			AHelper::Info<int>("DucttapeMergeDeformer error no geom it", 0);
			return status;
		}
		
		m_inputGeomIter = aiter;
	}
	
	if(!m_inputGeomIter) return status;
	
	MPoint pd;
	MVector dv;
	for (; !iter.isDone(); iter.next()) {
		if(m_inputGeomIter->isDone() ) return status;
		
		float wei = env * weightValue(block, multiIndex, iter.index() );
		if(wei > 1e-3f) {
			pd = iter.position();
			dv = m_inputGeomIter->position() - pd;
			pd = pd + dv * wei;
			iter.setPosition(pd);
		}
		
		m_inputGeomIter->next();
	}
	return status;
}