#include "ShrubVizNode.h"
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MVectorArray.h>
#include <maya/MFnVectorArrayData.h>
#include <maya/MFnMatrixData.h>
#include <maya/MFnPointArrayData.h>
#include <maya/MFnDoubleArrayData.h>
#include <maya/MFnIntArrayData.h>
#include <maya/MIntArray.h>
#include <maya/MPointArray.h>
#include <maya/MFnMeshData.h>
#include <AHelper.h>
#include <ExampVox.h>

namespace aphid {

MTypeId ShrubVizNode::id( 0x7809778 );
MObject ShrubVizNode::ashrubbox;
MObject ShrubVizNode::outValue;

ShrubVizNode::ShrubVizNode()
{ attachSceneCallbacks(); }

ShrubVizNode::~ShrubVizNode() 
{ detachSceneCallbacks(); }

MStatus ShrubVizNode::compute( const MPlug& plug, MDataBlock& block )
{
	if( plug == outValue ) {
		
		MDataHandle outputHandle = block.outputValue( outValue );
		outputHandle.set( 42 );
		block.setClean(plug);
    }

	return MS::kSuccess;
}

void ShrubVizNode::draw( M3dView & view, const MDagPath & path, 
							 M3dView::DisplayStyle style,
							 M3dView::DisplayStatus status )
{
	MObject thisNode = thisMObject();
		
	view.beginGL();
	
	glPushMatrix();
	
	BoundingBox bbox;
	getBBox(bbox);
	
	drawBoundingBox(&bbox);

	if ( style == M3dView::kFlatShaded || 
		    style == M3dView::kGouraudShaded ) {		
		
	}
	else {
		
	}
	
	glPopMatrix();
	
	view.endGL();
}

bool ShrubVizNode::isBounded() const
{ return true; }

MBoundingBox ShrubVizNode::boundingBox() const
{   
	BoundingBox bbox(-1,-1,-1,1,1,1);
	
	getBBox(bbox);
	
	MPoint corner1(bbox.getMin(0), bbox.getMin(1), bbox.getMin(2));
	MPoint corner2(bbox.getMax(0), bbox.getMax(1), bbox.getMax(2));

	return MBoundingBox( corner1, corner2 );
}

void* ShrubVizNode::creator()
{
	return new ShrubVizNode();
}

MStatus ShrubVizNode::initialize()
{ 
	MFnNumericAttribute numFn;
	MFnTypedAttribute typFn;
	MStatus			 stat;
	
	MDoubleArray defaultDArray;
	MFnDoubleArrayData dArrayDataFn;
	dArrayDataFn.create( defaultDArray );
	
	MVectorArray defaultVectArray;
	MFnVectorArrayData vectArrayDataFn;
	vectArrayDataFn.create( defaultVectArray );
	
	ashrubbox = typFn.create( "shrubBox", "sbbx",
									MFnData::kVectorArray, vectArrayDataFn.object(),
									&stat );
											
	if(stat != MS::kSuccess) {
		MGlobal::displayWarning("failed create shrub box attrib");
	}
	
	typFn.setStorable(true);
	
	stat = addAttribute(ashrubbox);
	if(stat != MS::kSuccess) {
		MGlobal::displayWarning("failed add shrub box attrib");
	}
		
    outValue = numFn.create( "outValue", "ov", MFnNumericData::kFloat );
	numFn.setStorable(false);
	numFn.setWritable(false);
	addAttribute(outValue);
	
	MPointArray defaultPntArray;
	MFnPointArrayData pntArrayDataFn;
	pntArrayDataFn.create( defaultPntArray );
	    
	attributeAffects(ashrubbox, outValue);
	
	return MS::kSuccess;
}

void ShrubVizNode::attachSceneCallbacks()
{
	fBeforeSaveCB  = MSceneMessage::addCallback(MSceneMessage::kBeforeSave,  releaseCallback, this);
}

void ShrubVizNode::detachSceneCallbacks()
{
	if (fBeforeSaveCB)
		MMessage::removeCallback(fBeforeSaveCB);

	fBeforeSaveCB = 0;
}

void ShrubVizNode::releaseCallback(void* clientData)
{
	ShrubVizNode *pThis = (ShrubVizNode*) clientData;
	pThis->saveInternal();
}

void ShrubVizNode::saveInternal()
{
	AHelper::Info<MString>("shrub save internal", MFnDependencyNode(thisMObject()).name() );
}

bool ShrubVizNode::loadInternal(MDataBlock& block)
{
	AHelper::Info<MString>("shrub load internal", MFnDependencyNode(thisMObject()).name() );
	return true;
}

MStatus ShrubVizNode::connectionMade ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	//if(plug == acameraspace) enableView();
	//AHelper::Info<MString>("connect", plug.name());
	return MPxLocatorNode::connectionMade (plug, otherPlug, asSrc );
}

MStatus ShrubVizNode::connectionBroken ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	//if(plug == acameraspace) disableView();
	//AHelper::Info<MString>("disconnect", plug.name());
	return MPxLocatorNode::connectionMade (plug, otherPlug, asSrc );
}

void ShrubVizNode::setBBox(const BoundingBox & bbox)
{
	MVectorArray dbox; dbox.setLength(2);
	dbox[0] = MVector(bbox.getMin(0), bbox.getMin(1), bbox.getMin(2) );
	dbox[1] = MVector(bbox.getMax(0), bbox.getMax(1), bbox.getMax(2) );
	MFnVectorArrayData vecFn;
	MObject obox = vecFn.create(dbox);
	MPlug dboxPlug(thisMObject(), ashrubbox );
	dboxPlug.setValue(obox);
}

void ShrubVizNode::getBBox(BoundingBox & bbox) const
{
	MPlug dboxPlug(thisMObject(), ashrubbox);
	MObject obox;
	dboxPlug.getValue(obox);
	
	MFnVectorArrayData vecFn(obox);
	MVectorArray dbox = vecFn.array();
	
	if(dbox.length() < 2) {
		AHelper::Info<unsigned>(" WARNING ShrubVizNode getBBox invalid data n", dbox.length() );
		return;
	}
	
	bbox.setMin(dbox[0].x, dbox[0].y, dbox[0].z );
	bbox.setMax(dbox[1].x, dbox[1].y, dbox[1].z );
}

}
//:~