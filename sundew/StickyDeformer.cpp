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
MObject StickyDeformer::ainVecX;
MObject StickyDeformer::ainVecY;
MObject StickyDeformer::ainVecZ;
MObject StickyDeformer::ainVec;

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
	
	ainVecX = numericFn.create("vecX", "vcx", 
										 MFnNumericData::kDouble, 0.0, &stat);
	ainVecY = numericFn.create("vecY", "vcy",
										 MFnNumericData::kDouble, 0.0, &stat);
	ainVecZ = numericFn.create("vecZ", "vcz",
										 MFnNumericData::kDouble, 0.0, &stat);
	ainVec = numericFn.create("inVec", "ivc",
										ainVecX,
										ainVecY,
										ainVecZ, &stat);
										
	numericFn.setStorable(false);
	numericFn.setWritable(true);
	addAttribute(ainVec);
    attributeAffects(ainVec, outputGeom);
	
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
	MDataHandle envData = block.inputValue(envelope,&status);
	const float env = envData.asFloat();
	if(env < 1e-3f) return status;
	
	MDataHandle vecData = block.inputValue(ainVec,&status);
	MVector displaceVec = vecData.asVector();
	if(displaceVec.length() < 1e-3) return status;
	// AHelper::Info<MVector>("def input displace", displaceVec);
	
	MDataHandle radiusData = block.inputValue(aradius,&status);
	double radius = radiusData.asDouble();
	if(radius < 1e-3) return status;
	// AHelper::Info<double>("def input radius", radius);
	
	MDataHandle meanData = block.inputValue(ainMean,&status);
	MVector mean = meanData.asVector();
	// AHelper::Info<MVector>("def input mean", mean);
	
	MPoint pt;
	MVector topt;
	double l, wei;
	for (; !iter.isDone(); iter.next()) {
		pt = iter.position();
		topt = pt - mean;
		l = topt.length();
		if(l < radius) {
			wei = 1.0 - l / radius;
			if(wei > 0.9) wei = 0.9;
			pt += displaceVec * (env * wei);
			iter.setPosition(pt);
		}
	}
	return status;
}
//:~
