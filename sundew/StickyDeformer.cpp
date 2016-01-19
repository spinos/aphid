#include "StickyDeformer.h"
#include <string.h>
#include <maya/MIOStream.h>
#include <math.h>
#include <fstream>
#include <maya/MItGeometry.h>
#include <boost/format.hpp>

#include <maya/MPlug.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MGlobal.h>
#include <maya/MPoint.h>
#include <maya/MMatrix.h>
#include <maya/MTime.h>
#include <maya/MFnMeshData.h>
#include <maya/MFnDagNode.h>
#include <maya/MFnMesh.h>
#include <maya/MPointArray.h>
#include <maya/MItMeshPolygon.h>
#include <maya/MProgressWindow.h>
#include <maya/MFnSingleIndexedComponent.h>
#include <maya/MFnPluginData.h>
#include <Mplg.h>
#include <AHelper.h>

MTypeId     StickyDeformer::id( 0xd76a847 );

MObject     StickyDeformer::ainMeanX;
MObject StickyDeformer::ainMeanY;
MObject StickyDeformer::ainMeanZ;
MObject StickyDeformer::ainMean;
MObject StickyDeformer::aradius;

StickyDeformer::StickyDeformer()
{}

StickyDeformer::~StickyDeformer()
{}

void* StickyDeformer::creator()
{
	return new StickyDeformer();
}

MStatus StickyDeformer::initialize()
{
	MStatus stat;

	MFnNumericAttribute numericFn;
	MFnTypedAttribute typedAttr;
	
	aradius = numericFn.create("radius", "rds", 
										 MFnNumericData::kDouble, 0.0, &stat);
	numericFn.setStorable(false);
	numericFn.setWritable(true);
	addAttribute(aradius);
    attributeAffects(aradius, outputGeom);
	
	ainMeanX = numericFn.create("meanX", "mnx", 
										 MFnNumericData::kDouble, 0.0, &stat);
	ainMeanY = numericFn.create("meanY", "mny",
										 MFnNumericData::kDouble, 0.0, &stat);
	ainMeanZ = numericFn.create("meanZ", "mnz",
										 MFnNumericData::kDouble, 0.0, &stat);
	ainMean = numericFn.create("inMean", "imn",
										ainMeanX,
										ainMeanY,
										ainMeanZ, &stat);
										
	numericFn.setStorable(false);
	numericFn.setWritable(true);
	addAttribute(ainMean);
    attributeAffects(ainMean, outputGeom);
	
	return MS::kSuccess;
}

MStatus StickyDeformer::connectionMade ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	MStatus result;
	if(plug == ainMean) {
		
	}
	return MPxDeformerNode::connectionMade (plug, otherPlug, asSrc );
}

MStatus StickyDeformer::connectionBroken ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	MStatus result;
	if(plug == ainMean) {
		
	}
	
	return MPxDeformerNode::connectionBroken ( plug, otherPlug, asSrc );
}

MStatus StickyDeformer::deform( MDataBlock& block,
				MItGeometry& iter,
				const MMatrix& m,
				unsigned int multiIndex)
//
// Method: deform
//
// Description:   Deform the point with a StickyDeformer algorithm
//
// Arguments:
//   block		: the datablock of the node
//	 iter		: an iterator for the geometry to be deformed
//   m    		: matrix to transform the point into world space
//	 multiIndex : the index of the geometry that we are deforming
//
//
{
	MStatus status;

	// determine the envelope (this is a global scale factor)
	//
	MDataHandle envData = block.inputValue(envelope,&status);
	const float env = envData.asFloat();
	if(env < 0.001) return status;
	
	return status;
}
//:~
