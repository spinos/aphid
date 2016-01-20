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
MObject StickyDeformer::aradius;
MObject StickyDeformer::ainVecX;
MObject StickyDeformer::ainVecY;
MObject StickyDeformer::ainVecZ;
MObject StickyDeformer::ainVec;
MObject StickyDeformer::avertexSpace;
MObject StickyDeformer::adropoff;

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
	
	adropoff = numericFn.create("dropoff", "dpo", 
										 MFnNumericData::kDouble, 1.0, &stat);
	numericFn.setStorable(false);
	numericFn.setWritable(true);
	addAttribute(adropoff);
    attributeAffects(adropoff, outputGeom);
	
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
	
	MFnMatrixAttribute matAttr;
	avertexSpace = matAttr.create( "vertexMatrix", "vtm", MFnMatrixAttribute::kDouble );
 	matAttr.setStorable(false);
	matAttr.setWritable(true);
	matAttr.setConnectable(true);
	addAttribute(avertexSpace);
	attributeAffects(avertexSpace, outputGeom);
	MGlobal::executeCommand( "makePaintable -attrType multiFloat -sm deformer stickyDeformer weights" );
	return MS::kSuccess;
}

MStatus StickyDeformer::connectionMade ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	return MPxDeformerNode::connectionMade (plug, otherPlug, asSrc );
}

MStatus StickyDeformer::connectionBroken ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	return MPxDeformerNode::connectionBroken ( plug, otherPlug, asSrc );
}

MStatus StickyDeformer::deform( MDataBlock& block,
				MItGeometry& iter,
				const MMatrix& m,
				unsigned int multiIndex)
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
	
	MDataHandle dropoffData = block.inputValue(adropoff,&status);
	double dropoff = dropoffData.asDouble();
	
	MDataHandle rotData = block.inputValue(avertexSpace,&status);
	
	MMatrix rot = rotData.asMatrix();
	MVector mean(rot[3][0], rot[3][1], rot[3][2]);
	
	MVector worldDisplaceVec = MPoint(displaceVec) * rot - mean;
	
	MPoint pt;
	MVector topt;
	double l, wei;
	for (; !iter.isDone(); iter.next()) {
		pt = iter.position();
		topt = pt - mean;
		l = topt.length();
		if(l < radius) {
			wei = 1.0 - l / radius;
			wei = pow(wei, dropoff);
			if(wei > 0.93) wei = 0.93;
			pt += worldDisplaceVec * (env * wei * weightValue(block, multiIndex, iter.index() ) );
			iter.setPosition(pt);
		}
	}
	return status;
}
//:~
