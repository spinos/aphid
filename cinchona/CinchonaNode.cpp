#include "CinchonaNode.h"
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnVectorArrayData.h>
#include <maya/MFnMatrixData.h>
#include <maya/MFnPointArrayData.h>
#include <maya/MFnDoubleArrayData.h>
#include <maya/MFnIntArrayData.h>
#include <maya/MPointArray.h>
#include <maya/MFnMeshData.h>
#include <maya/MFnPluginData.h>
#include <AHelper.h>
#include <math/linearMath.h>
#include <mama/AttributeHelper.h>
#include <ogl/GlslInstancer.h>

using namespace aphid;

MTypeId CinchonaNode::id( 0xe5a8456 );
MObject CinchonaNode::ahumerusmat;
MObject CinchonaNode::aulnamat;
MObject CinchonaNode::aradiusmat;
MObject CinchonaNode::acarpusmat;
MObject CinchonaNode::aseconddigitmat;	
MObject CinchonaNode::aligament0x;
MObject CinchonaNode::aligament0y;
MObject CinchonaNode::aligament0z;
MObject CinchonaNode::aligament0;
MObject CinchonaNode::aligament1x;
MObject CinchonaNode::aligament1y;
MObject CinchonaNode::aligament1z;
MObject CinchonaNode::aligament1;
MObject CinchonaNode::aelbowos1x;
MObject CinchonaNode::aelbowos1y;
MObject CinchonaNode::aelbowos1z;
MObject CinchonaNode::aelbowos1;
MObject CinchonaNode::awristos0x;
MObject CinchonaNode::awristos0y;
MObject CinchonaNode::awristos0z;
MObject CinchonaNode::awristos0;
MObject CinchonaNode::awristos1x;
MObject CinchonaNode::awristos1y;
MObject CinchonaNode::awristos1z;
MObject CinchonaNode::awristos1;
MObject CinchonaNode::adigitos0x;
MObject CinchonaNode::adigitos0y;
MObject CinchonaNode::adigitos0z;
MObject CinchonaNode::adigitos0;
MObject CinchonaNode::adigitos1x;
MObject CinchonaNode::adigitos1y;
MObject CinchonaNode::adigitos1z;
MObject CinchonaNode::adigitos1;
MObject CinchonaNode::adigitl;
MObject CinchonaNode::outValue;
	
CinchonaNode::CinchonaNode()
{ attachSceneCallbacks(); }

CinchonaNode::~CinchonaNode() 
{ detachSceneCallbacks(); }

MStatus CinchonaNode::compute( const MPlug& plug, MDataBlock& block )
{
	if( plug == outValue ) {
		
		MDataHandle outputHandle = block.outputValue( outValue );
		outputHandle.set( 42 );
		block.setClean(plug);
    }

	return MS::kSuccess;
}

void CinchonaNode::draw( M3dView & view, const MDagPath & path, 
							 M3dView::DisplayStyle style,
							 M3dView::DisplayStatus status )
{	
	MObject thisNode = thisMObject();
	setSkeletonMatrices(thisNode, ahumerusmat, aulnamat, aradiusmat, acarpusmat, aseconddigitmat);
	updatePrincipleMatrix();
	setLigamentParams(thisNode, aligament0x, aligament0y, aligament0z,
					aligament1x, aligament1y, aligament1z);
	setElbowParams(thisNode, aelbowos1x, aelbowos1y, aelbowos1z);
	setWristParams(thisNode, awristos0x, awristos0y, awristos0z,
					awristos1x, awristos1y, awristos1z);
	set2ndDigitParams(thisNode, adigitos0x, adigitos0y, adigitos0z,
					adigitos1x, adigitos1y, adigitos1z,
					adigitl);
					
	updateLigaments();
	
	view.beginGL();
	
	drawSkeletonCoordinates();
	drawLigaments();
	
#if 0
	bool hasGlsl = isGlslReady();
	if(!hasGlsl ) {
		hasGlsl = prepareGlsl();
	}
#else
	bool hasGlsl = false;
#endif
	
	if(hasGlsl ) {

/// https://www.opengl.org/sdk/docs/man2/xhtml/glPushAttrib.xml	
	glPushAttrib(GL_ALL_ATTRIB_BITS);
		
	if ( style == M3dView::kFlatShaded || 
		    style == M3dView::kGouraudShaded ) {

	}
	else {
	
	}
	
	glPopAttrib();
	
	} else {
		//AHelper::Info<std::string >(" ERROR opengl ", "has no glsl");
	}
	
	view.endGL();
}

bool CinchonaNode::isBounded() const
{ return false; }

MBoundingBox CinchonaNode::boundingBox() const
{   
	BoundingBox bbox(-1,-1,-1,1,1,1);
	
	MPoint corner1(bbox.getMin(0), bbox.getMin(1), bbox.getMin(2));
	MPoint corner2(bbox.getMax(0), bbox.getMax(1), bbox.getMax(2));

	return MBoundingBox( corner1, corner2 );
}

void* CinchonaNode::creator()
{
	return new CinchonaNode();
}

MStatus CinchonaNode::initialize()
{ 
	MFnNumericAttribute numFn;
	MFnTypedAttribute typFn;
	MStatus			 stat;
	
	
	MFnMatrixAttribute matAttr;
	
	AttributeHelper::CreateVector3FAttrib(aligament0, aligament0x, aligament0y, aligament0z,
		"shoulderos0", "sd0", 0.f, 0.f, 1.f);
	addAttribute(aligament0x);
	addAttribute(aligament0y);
	addAttribute(aligament0z);
	addAttribute(aligament0); 
	
	AttributeHelper::CreateVector3FAttrib(aligament1, aligament1x, aligament1y, aligament1z,
		"shoulderos1", "sd1", 0.f, 0.f,-1.f);
	addAttribute(aligament1x);
	addAttribute(aligament1y);
	addAttribute(aligament1z);
	addAttribute(aligament1);
	
	AttributeHelper::CreateVector3FAttrib(aelbowos1, aelbowos1x, aelbowos1y, aelbowos1z,
		"elbowos1", "eb1", 0.f, 0.f,-1.f);
	addAttribute(aelbowos1x);
	addAttribute(aelbowos1y);
	addAttribute(aelbowos1z);
	addAttribute(aelbowos1); 
	
	AttributeHelper::CreateVector3FAttrib(awristos0, awristos0x, awristos0y, awristos0z,
		"wristos0", "ws0", 0.f, 0.f, 1.f);
	addAttribute(awristos0x);
	addAttribute(awristos0y);
	addAttribute(awristos0z);
	addAttribute(awristos0);
	
	AttributeHelper::CreateVector3FAttrib(awristos1, awristos1x, awristos1y, awristos1z,
		"wristos1", "ws1", 0.f, 0.f,-1.f);
	addAttribute(awristos1x);
	addAttribute(awristos1y);
	addAttribute(awristos1z);
	addAttribute(awristos1);
	
	AttributeHelper::CreateVector3FAttrib(adigitos0, adigitos0x, adigitos0y, adigitos0z,
		"snddigos0", "ds0", 0.f, 0.f, .7f);
	addAttribute(adigitos0x);
	addAttribute(adigitos0y);
	addAttribute(adigitos0z);
	addAttribute(adigitos0);
	
	AttributeHelper::CreateVector3FAttrib(adigitos1, adigitos1x, adigitos1y, adigitos1z,
		"snddigos1", "ds1", 0.f, 0.f, -.7f);
	addAttribute(adigitos1x);
	addAttribute(adigitos1y);
	addAttribute(adigitos1z);
	addAttribute(adigitos1);
	
	adigitl = numFn.create( "snddiglen", "dgl", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(2.f);
	addAttribute(adigitl);
	
	ahumerusmat = matAttr.create("humerusMatrix", "hsm", MFnMatrixAttribute::kDouble);
	matAttr.setStorable(false);
	matAttr.setConnectable(true);
	addAttribute(ahumerusmat);
	
	aulnamat = matAttr.create("ulnaMatrix", "uam", MFnMatrixAttribute::kDouble);
	matAttr.setStorable(false);
	matAttr.setConnectable(true);
	addAttribute(aulnamat);
	
	aradiusmat = matAttr.create("radiusMatrix", "rsm", MFnMatrixAttribute::kDouble);
	matAttr.setStorable(false);
	matAttr.setConnectable(true);
	addAttribute(aradiusmat);
	
	acarpusmat = matAttr.create("carpusMatrix", "cpm", MFnMatrixAttribute::kDouble);
	matAttr.setStorable(false);
	matAttr.setConnectable(true);
	addAttribute(acarpusmat);
	
	aseconddigitmat = matAttr.create("secondDigitMatrix", "sdm", MFnMatrixAttribute::kDouble);
	matAttr.setStorable(false);
	matAttr.setConnectable(true);
	addAttribute(aseconddigitmat);
		
    outValue = numFn.create( "outValue", "ov", MFnNumericData::kFloat );
	numFn.setStorable(false);
	numFn.setWritable(false);
	addAttribute(outValue);
	
	MPointArray defaultPntArray;
	MFnPointArrayData pntArrayDataFn;
	pntArrayDataFn.create( defaultPntArray );
	    
	attributeAffects(aligament0x, outValue);
	attributeAffects(aligament0y, outValue);
	attributeAffects(aligament0z, outValue);
	attributeAffects(aligament1x, outValue);
	attributeAffects(aligament1y, outValue);
	attributeAffects(aligament1z, outValue);
	attributeAffects(aelbowos1x, outValue);
	attributeAffects(aelbowos1y, outValue);
	attributeAffects(aelbowos1z, outValue);
	attributeAffects(awristos0x, outValue);
	attributeAffects(awristos0y, outValue);
	attributeAffects(awristos0z, outValue);
	attributeAffects(awristos1x, outValue);
	attributeAffects(awristos1y, outValue);
	attributeAffects(awristos1z, outValue);
	attributeAffects(adigitos0x, outValue);
	attributeAffects(adigitos0y, outValue);
	attributeAffects(adigitos0z, outValue);
	attributeAffects(adigitos1x, outValue);
	attributeAffects(adigitos1y, outValue);
	attributeAffects(adigitos1z, outValue);
	attributeAffects(adigitl, outValue);
	attributeAffects(ahumerusmat, outValue);
	attributeAffects(aulnamat, outValue);
	attributeAffects(aradiusmat, outValue);
	
	return MS::kSuccess;
}

void CinchonaNode::attachSceneCallbacks()
{
	fBeforeSaveCB  = MSceneMessage::addCallback(MSceneMessage::kBeforeSave,  releaseCallback, this);
}

void CinchonaNode::detachSceneCallbacks()
{
	if (fBeforeSaveCB)
		MMessage::removeCallback(fBeforeSaveCB);

	fBeforeSaveCB = 0;
}

void CinchonaNode::releaseCallback(void* clientData)
{
	CinchonaNode *pThis = (CinchonaNode*) clientData;
	pThis->saveInternal();
}

void CinchonaNode::saveInternal()
{
	AHelper::Info<MString>("cinchona save internal", MFnDependencyNode(thisMObject()).name() );
	
}

bool CinchonaNode::loadInternal()
{
	AHelper::Info<MString>("cinchona load internal", MFnDependencyNode(thisMObject()).name() );
	
	return true;
}

bool CinchonaNode::loadInternal(MDataBlock& block)
{
	AHelper::Info<MString>("cinchona load internal", MFnDependencyNode(thisMObject()).name() );

	return true;
}

MStatus CinchonaNode::connectionMade ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	if(plug == ahumerusmat) {
		
	}
	return MPxLocatorNode::connectionMade (plug, otherPlug, asSrc );
}

MStatus CinchonaNode::connectionBroken ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	if(plug == ahumerusmat) {
		AHelper::Info<MString>("disconnect", plug.name());
	}
	return MPxLocatorNode::connectionMade (plug, otherPlug, asSrc );
}
//:~