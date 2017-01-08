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
#include <mama/AttributeHelper.h>
#include <ogl/GlslInstancer.h>

using namespace aphid;

MTypeId CinchonaNode::id( 0xe5a8456 );
MObject CinchonaNode::ahumerusmat;
MObject CinchonaNode::aulnamat;
MObject CinchonaNode::aradiusmat;
MObject CinchonaNode::acarpusmat;
MObject CinchonaNode::aseconddigitmat;
MObject CinchonaNode::anumfeather0;
MObject CinchonaNode::anumfeather1;
MObject CinchonaNode::anumfeather2;
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
MObject CinchonaNode::adigitl;
MObject CinchonaNode::ainboardmat;
MObject CinchonaNode::amidsectmat0;
MObject CinchonaNode::amidsectmat1;
MObject CinchonaNode::achord0;
MObject CinchonaNode::achord1;
MObject CinchonaNode::achord2;
MObject CinchonaNode::achord3;
MObject CinchonaNode::athickness0;
MObject CinchonaNode::athickness1;
MObject CinchonaNode::athickness2;
MObject CinchonaNode::athickness3;
MObject CinchonaNode::abrt0mat;
MObject CinchonaNode::abrt1mat;
MObject CinchonaNode::abrt2mat;
MObject CinchonaNode::abrt3mat;
MObject CinchonaNode::aup0n0;
MObject CinchonaNode::aup0n1;
MObject CinchonaNode::aup0n2;
MObject CinchonaNode::aup0n3;
MObject CinchonaNode::aup0c0;
MObject CinchonaNode::aup0c1;
MObject CinchonaNode::aup0c2;
MObject CinchonaNode::aup0c3;
MObject CinchonaNode::aup0c4;
MObject CinchonaNode::aup0t0;
MObject CinchonaNode::aup0t1;
MObject CinchonaNode::aup0t2;
MObject CinchonaNode::aup0t3;
MObject CinchonaNode::aup0t4;
MObject CinchonaNode::aup0rz;
MObject CinchonaNode::aup1n0;
MObject CinchonaNode::aup1n1;
MObject CinchonaNode::aup1n2;
MObject CinchonaNode::aup1n3;
MObject CinchonaNode::aup1c0;
MObject CinchonaNode::aup1c1;
MObject CinchonaNode::aup1c2;
MObject CinchonaNode::aup1c3;
MObject CinchonaNode::aup1c4;
MObject CinchonaNode::aup1t0;
MObject CinchonaNode::aup1t1;
MObject CinchonaNode::aup1t2;
MObject CinchonaNode::aup1t3;
MObject CinchonaNode::aup1t4;
MObject CinchonaNode::aup1rz;
MObject CinchonaNode::alow0n0;
MObject CinchonaNode::alow0n1;
MObject CinchonaNode::alow0n2;
MObject CinchonaNode::alow0n3;
MObject CinchonaNode::alow0c0;
MObject CinchonaNode::alow0c1;
MObject CinchonaNode::alow0c2;
MObject CinchonaNode::alow0c3;
MObject CinchonaNode::alow0c4;
MObject CinchonaNode::alow0t0;
MObject CinchonaNode::alow0t1;
MObject CinchonaNode::alow0t2;
MObject CinchonaNode::alow0t3;
MObject CinchonaNode::alow0t4;
MObject CinchonaNode::alow0rz;
MObject CinchonaNode::alow1n0;
MObject CinchonaNode::alow1n1;
MObject CinchonaNode::alow1n2;
MObject CinchonaNode::alow1n3;
MObject CinchonaNode::alow1c0;
MObject CinchonaNode::alow1c1;
MObject CinchonaNode::alow1c2;
MObject CinchonaNode::alow1c3;
MObject CinchonaNode::alow1c4;
MObject CinchonaNode::alow1t0;
MObject CinchonaNode::alow1t1;
MObject CinchonaNode::alow1t2;
MObject CinchonaNode::alow1t3;
MObject CinchonaNode::alow1t4;
MObject CinchonaNode::alow1rz;
MObject CinchonaNode::ayawnoi;
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
	updateHandMatrix();
	updateFingerMatrix();
	setLigamentParams(thisNode, aligament0x, aligament0y, aligament0z,
					aligament1x, aligament1y, aligament1z);
	setElbowParams(thisNode, aelbowos1x, aelbowos1y, aelbowos1z);
	setWristParams(thisNode, awristos0x, awristos0y, awristos0z,
					awristos1x, awristos1y, awristos1z);
	set2ndDigitParams(thisNode, adigitos0x, adigitos0y, adigitos0z,
					adigitl);
	setFirstLeadingLigament();
	updateLigaments();
	setFlyingFeatherGeomParam(thisNode, anumfeather0, anumfeather1, anumfeather2,
						achord0, achord1, achord2, achord3,
						athickness0, athickness1, athickness2, athickness3);
	setCovertFeatherGeomParam(1, thisNode, aup0n0, aup0n1, aup0n2, aup0n3,
						aup0c0, aup0c1, aup0c2, aup0c3, aup0c4,
						aup0t0, aup0t1, aup0t2, aup0t3, aup0t4);
	setCovertFeatherGeomParam(2, thisNode, aup1n0, aup1n1, aup1n2, aup1n3,
						aup1c0, aup1c1, aup1c2, aup1c3, aup1c4,
						aup1t0, aup1t1, aup1t2, aup1t3, aup1t4);
	setCovertFeatherGeomParam(3, thisNode, alow0n0, alow0n1, alow0n2, alow0n3,
						alow0c0, alow0c1, alow0c2, alow0c3, alow0c4,
						alow0t0, alow0t1, alow0t2, alow0t3, alow0t4);
	setCovertFeatherGeomParam(4, thisNode, alow1n0, alow1n1, alow1n2, alow1n3,
						alow1c0, alow1c1, alow1c2, alow1c3, alow1c4,
						alow1t0, alow1t1, alow1t2, alow1t3, alow1t4);
						
	updateFeatherGeom();
	setFeatherOrientationParam(thisNode, ainboardmat, amidsectmat0, amidsectmat1,
						aup0rz, aup1rz, alow0rz, alow1rz,
						ayawnoi);
	updateFeatherTransform();
	setFeatherDeformationParam(thisNode, abrt0mat, abrt1mat, abrt2mat, abrt3mat);
	updateFeatherDeformation();
	
	view.beginGL();
	
	drawFeatherLeadingEdges();
	drawFeatherContours();
	drawSkeletonCoordinates();
	drawLigaments();
	drawFeatherOrietations();
	drawRibs();
	drawSpars();
	
#if 0
	bool hasGlsl = isGlslReady();
	if(!hasGlsl ) {
		hasGlsl = prepareGlsl();
	}
	bool hasGlsl = false;
#endif

#if 0
	if(hasGlsl ) {
#endif

/// https://www.opengl.org/sdk/docs/man2/xhtml/glPushAttrib.xml	
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	float colf[] = {.5f, .6f, .7f};
	float colb[] = {.9f, .9f, 0.f};
	glMaterialfv(GL_FRONT, GL_DIFFUSE, colf);
	glMaterialfv(GL_BACK, GL_DIFFUSE, colb);
	glEnable(GL_LIGHTING);
	drawFeathers();
		
	if ( style == M3dView::kFlatShaded || 
		    style == M3dView::kGouraudShaded ) {

	}
	else {
	
	}
	
	glPopAttrib();
	
#if 0
	} else {
		AHelper::Info<std::string >(" ERROR opengl ", "has no glsl");
	}
#endif

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
	MFnMatrixAttribute matAttr;
	MStatus			 stat;
	
	anumfeather0 = numFn.create( "numFeather0", "nfr0", MFnNumericData::kInt);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(7);
	numFn.setMin(2);
	numFn.setMax(50);
	addAttribute(anumfeather0);
	
	anumfeather1 = numFn.create( "numFeather1", "nfr1", MFnNumericData::kInt);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(7);
	numFn.setMin(2);
	numFn.setMax(50);
	addAttribute(anumfeather1);
	
	anumfeather2 = numFn.create( "numFeather2", "nfr2", MFnNumericData::kInt);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(7);
	numFn.setMin(2);
	numFn.setMax(50);
	addAttribute(anumfeather2);
	
	achord0 = numFn.create( "chord0", "chd0", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(20.f);
	numFn.setMin(1.f);
	addAttribute(achord0);
	
	achord1 = numFn.create( "chord1", "chd1", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(20.f);
	numFn.setMin(1.f);
	addAttribute(achord1);
	
	achord2 = numFn.create( "chord2", "chd2", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(20.f);
	numFn.setMin(1);
	addAttribute(achord2);
	
	achord3 = numFn.create( "chord3", "chd3", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(20.f);
	numFn.setMin(1.f);
	addAttribute(achord3);
	
	athickness0 = numFn.create( "thickness0", "thk0", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.15f);
	numFn.setMin(0.01f);
	addAttribute(athickness0);
	
	athickness1 = numFn.create( "thickness1", "thk1", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.15f);
	numFn.setMin(0.01f);
	addAttribute(athickness1);
	
	athickness2 = numFn.create( "thickness2", "thk2", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.15f);
	numFn.setMin(0.01f);
	addAttribute(athickness2);
	
	athickness3 = numFn.create( "thickness3", "thk3", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.15f);
	numFn.setMin(0.01f);
	addAttribute(athickness3);
	
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
	
	adigitl = numFn.create( "snddiglen", "dgl", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(5.f);
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
	
	ainboardmat = matAttr.create("inboardMatrix", "ibm", MFnMatrixAttribute::kDouble);
	matAttr.setStorable(false);
	matAttr.setConnectable(true);
	addAttribute(ainboardmat);
	
	amidsectmat0 = matAttr.create("midsection0Matrix", "ms0m", MFnMatrixAttribute::kDouble);
	matAttr.setStorable(false);
	matAttr.setConnectable(true);
	addAttribute(amidsectmat0);
	
	amidsectmat1 = matAttr.create("midsection1Matrix", "ms1m", MFnMatrixAttribute::kDouble);
	matAttr.setStorable(false);
	matAttr.setConnectable(true);
	addAttribute(amidsectmat1);

	abrt0mat = matAttr.create("bendRollTwist0Matrix", "brt0m", MFnMatrixAttribute::kDouble);
	matAttr.setStorable(false);
	matAttr.setConnectable(true);
	addAttribute(abrt0mat);
	
	abrt1mat = matAttr.create("bendRollTwist1Matrix", "brt1m", MFnMatrixAttribute::kDouble);
	matAttr.setStorable(false);
	matAttr.setConnectable(true);
	addAttribute(abrt1mat);
	
	abrt2mat = matAttr.create("bendRollTwist2Matrix", "brt2m", MFnMatrixAttribute::kDouble);
	matAttr.setStorable(false);
	matAttr.setConnectable(true);
	addAttribute(abrt2mat);
	
	abrt3mat = matAttr.create("bendRollTwist3Matrix", "brt3m", MFnMatrixAttribute::kDouble);
	matAttr.setStorable(false);
	matAttr.setConnectable(true);
	addAttribute(abrt3mat);
	
	aup0n0 = numFn.create( "up0NumFeather0", "u0n0", MFnNumericData::kInt);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(7);
	numFn.setMin(2);
	numFn.setMax(50);
	addAttribute(aup0n0);
	
	aup0n1 = numFn.create( "up0NumFeather1", "u0n1", MFnNumericData::kInt);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(9);
	numFn.setMin(2);
	numFn.setMax(50);
	addAttribute(aup0n1);
	
	aup0n2 = numFn.create( "up0NumFeather2", "u0n2", MFnNumericData::kInt);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(4);
	numFn.setMin(2);
	numFn.setMax(50);
	addAttribute(aup0n2);
	
	aup0n3 = numFn.create( "up0NumFeather3", "u0n3", MFnNumericData::kInt);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(10);
	numFn.setMin(2);
	numFn.setMax(50);
	addAttribute(aup0n3);
	
	aup0c0 = numFn.create( "up0Chord0", "u0c0", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(10.f);
	numFn.setMin(0.5f);
	addAttribute(aup0c0);
	
	aup0c1 = numFn.create( "up0Chord1", "u0c1", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(8.f);
	numFn.setMin(0.5f);
	addAttribute(aup0c1);
	
	aup0c2 = numFn.create( "up0Chord2", "u0c2", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(7.f);
	numFn.setMin(0.5f);
	addAttribute(aup0c2);
	
	aup0c3 = numFn.create( "up0Chord3", "u0c3", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(8.f);
	numFn.setMin(0.5f);
	addAttribute(aup0c3);
	
	aup0c4 = numFn.create( "up0Chord4", "u0c4", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(12.f);
	numFn.setMin(0.5f);
	addAttribute(aup0c4);
	
	aup0t0 = numFn.create( "up0Thickness0", "u0t0", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.2f);
	numFn.setMin(0.01f);
	addAttribute(aup0t0);
	
	aup0t1 = numFn.create( "up0Thickness1", "u0t1", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.2f);
	numFn.setMin(0.01f);
	addAttribute(aup0t1);
	
	aup0t2 = numFn.create( "up0Thickness2", "u0t2", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.19f);
	numFn.setMin(0.01f);
	addAttribute(aup0t2);
	
	aup0t3 = numFn.create( "up0Thickness3", "u0t3", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.19f);
	numFn.setMin(0.01f);
	addAttribute(aup0t3);
	
	aup0t4 = numFn.create( "up0Thickness4", "u0t4", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.17f);
	numFn.setMin(0.01f);
	addAttribute(aup0t4);
	
		aup1n0 = numFn.create( "up1NumFeather0", "u1n0", MFnNumericData::kInt);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(10);
	numFn.setMin(2);
	numFn.setMax(50);
	addAttribute(aup1n0);
	
	aup1n1 = numFn.create( "up1NumFeather1", "u1n1", MFnNumericData::kInt);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(11);
	numFn.setMin(2);
	numFn.setMax(50);
	addAttribute(aup1n1);
	
	aup1n2 = numFn.create( "up1NumFeather2", "u1n2", MFnNumericData::kInt);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(6);
	numFn.setMin(2);
	numFn.setMax(50);
	addAttribute(aup1n2);
	
	aup1n3 = numFn.create( "up1NumFeather3", "u1n3", MFnNumericData::kInt);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(6);
	numFn.setMin(2);
	numFn.setMax(50);
	addAttribute(aup1n3);
	
	aup1c0 = numFn.create( "up1Chord0", "u1c0", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(9.f);
	numFn.setMin(0.5f);
	addAttribute(aup1c0);
	
	aup1c1 = numFn.create( "up1Chord1", "u1c1", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(9.f);
	numFn.setMin(0.5f);
	addAttribute(aup1c1);
	
	aup1c2 = numFn.create( "up1Chord2", "u1c2", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(7.f);
	numFn.setMin(0.5f);
	addAttribute(aup1c2);
	
	aup1c3 = numFn.create( "up1Chord3", "u1c3", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(6.f);
	numFn.setMin(0.5f);
	addAttribute(aup1c3);
	
	aup1c4 = numFn.create( "up1Chord4", "u1c4", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(6.f);
	numFn.setMin(0.5f);
	addAttribute(aup1c4);
	
	aup1t0 = numFn.create( "up1Thickness0", "u1t0", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.23f);
	numFn.setMin(0.01f);
	addAttribute(aup1t0);
	
	aup1t1 = numFn.create( "up1Thickness1", "u1t1", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.3f);
	numFn.setMin(0.01f);
	addAttribute(aup1t1);
	
	aup1t2 = numFn.create( "up1Thickness2", "u1t2", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.29f);
	numFn.setMin(0.01f);
	addAttribute(aup1t2);
	
	aup1t3 = numFn.create( "up1Thickness3", "u1t3", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.29f);
	numFn.setMin(0.01f);
	addAttribute(aup1t3);
	
	aup1t4 = numFn.create( "up1Thickness4", "u1t4", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.27f);
	numFn.setMin(0.01f);
	addAttribute(aup1t4);
	
	alow0n0 = numFn.create( "low0NumFeather0", "l0n0", MFnNumericData::kInt);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(7);
	numFn.setMin(2);
	numFn.setMax(50);
	addAttribute(alow0n0);
	
	alow0n1 = numFn.create( "low0NumFeather1", "l0n1", MFnNumericData::kInt);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(9);
	numFn.setMin(2);
	numFn.setMax(50);
	addAttribute(alow0n1);
	
	alow0n2 = numFn.create( "low0NumFeather2", "l0n2", MFnNumericData::kInt);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(4);
	numFn.setMin(2);
	numFn.setMax(50);
	addAttribute(alow0n2);
	
	alow0n3 = numFn.create( "low0NumFeather3", "l0n3", MFnNumericData::kInt);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(10);
	numFn.setMin(2);
	numFn.setMax(50);
	addAttribute(alow0n3);
	
	alow0c0 = numFn.create( "low0Chord0", "l0c0", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(10.f);
	numFn.setMin(0.5f);
	addAttribute(alow0c0);
	
	alow0c1 = numFn.create( "low0Chord1", "l0c1", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(10.f);
	numFn.setMin(0.5f);
	addAttribute(alow0c1);
	
	alow0c2 = numFn.create( "low0Chord2", "l0c2", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(10.f);
	numFn.setMin(0.5f);
	addAttribute(alow0c2);
	
	alow0c3 = numFn.create( "low0Chord3", "l0c3", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(10.f);
	numFn.setMin(0.5f);
	addAttribute(alow0c3);
	
	alow0c4 = numFn.create( "low0Chord4", "l0c4", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(12.f);
	numFn.setMin(0.5f);
	addAttribute(alow0c4);
	
	alow0t0 = numFn.create( "low0Thickness0", "l0t0", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.2f);
	numFn.setMin(0.01f);
	addAttribute(alow0t0);
	
	alow0t1 = numFn.create( "low0Thickness1", "l0t1", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.2f);
	numFn.setMin(0.01f);
	addAttribute(alow0t1);
	
	alow0t2 = numFn.create( "low0Thickness2", "l0t2", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.19f);
	numFn.setMin(0.01f);
	addAttribute(alow0t2);
	
	alow0t3 = numFn.create( "low0Thickness3", "l0t3", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.19f);
	numFn.setMin(0.01f);
	addAttribute(alow0t3);
	
	alow0t4 = numFn.create( "low0Thickness4", "l0t4", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.17f);
	numFn.setMin(0.01f);
	addAttribute(alow0t4);
	
		alow1n0 = numFn.create( "low1NumFeather0", "l1n0", MFnNumericData::kInt);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(10);
	numFn.setMin(2);
	numFn.setMax(50);
	addAttribute(alow1n0);
	
	alow1n1 = numFn.create( "low1NumFeather1", "l1n1", MFnNumericData::kInt);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(11);
	numFn.setMin(2);
	numFn.setMax(50);
	addAttribute(alow1n1);
	
	alow1n2 = numFn.create( "low1NumFeather2", "l1n2", MFnNumericData::kInt);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(6);
	numFn.setMin(2);
	numFn.setMax(50);
	addAttribute(alow1n2);
	
	alow1n3 = numFn.create( "low1NumFeather3", "l1n3", MFnNumericData::kInt);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(6);
	numFn.setMin(2);
	numFn.setMax(50);
	addAttribute(alow1n3);
	
	alow1c0 = numFn.create( "low1Chord0", "l1c0", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(7.f);
	numFn.setMin(0.5f);
	addAttribute(alow1c0);
	
	alow1c1 = numFn.create( "low1Chord1", "l1c1", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(8.f);
	numFn.setMin(0.5f);
	addAttribute(alow1c1);
	
	alow1c2 = numFn.create( "low1Chord2", "l1c2", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(5.f);
	numFn.setMin(0.5f);
	addAttribute(alow1c2);
	
	alow1c3 = numFn.create( "low1Chord3", "l1c3", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(4.f);
	numFn.setMin(0.5f);
	addAttribute(alow1c3);
	
	alow1c4 = numFn.create( "low1Chord4", "l1c4", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(3.f);
	numFn.setMin(0.5f);
	addAttribute(alow1c4);
	
	alow1t0 = numFn.create( "low1Thickness0", "l1t0", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.23f);
	numFn.setMin(0.01f);
	addAttribute(alow1t0);
	
	alow1t1 = numFn.create( "low1Thickness1", "l1t1", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.3f);
	numFn.setMin(0.01f);
	addAttribute(alow1t1);
	
	alow1t2 = numFn.create( "low1Thickness2", "l1t2", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.29f);
	numFn.setMin(0.01f);
	addAttribute(alow1t2);
	
	alow1t3 = numFn.create( "low1Thickness3", "l1t3", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.29f);
	numFn.setMin(0.01f);
	addAttribute(alow1t3);
	
	alow1t4 = numFn.create( "low1Thickness4", "l1t4", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.27f);
	numFn.setMin(0.01f);
	addAttribute(alow1t4);
	
	aup0rz = numFn.create( "up0Pitch", "u0ph", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(.25f);
	numFn.setMin(0.f);
	numFn.setMax(10.f);
	addAttribute(aup0rz);
	
	aup1rz = numFn.create( "up1Pitch", "u1ph", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(.25f);
	numFn.setMin(0.f);
	numFn.setMax(10.f);
	addAttribute(aup1rz);
	
	alow0rz = numFn.create( "low0Pitch", "l0ph", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(.25f);
	numFn.setMin(0.f);
	numFn.setMax(10.f);
	addAttribute(alow0rz);
	
	alow1rz = numFn.create( "low1Pitch", "l1ph", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(.25f);
	numFn.setMin(0.f);
	numFn.setMax(10.f);
	addAttribute(alow1rz);

	ayawnoi = numFn.create( "yawNoise", "ywns", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.f);
	numFn.setMin(0.f);
	numFn.setMax(10.f);
	addAttribute(ayawnoi);
	
    outValue = numFn.create( "outValue", "ov", MFnNumericData::kFloat );
	numFn.setStorable(false);
	numFn.setWritable(false);
	addAttribute(outValue);
	
	MPointArray defaultPntArray;
	MFnPointArrayData pntArrayDataFn;
	pntArrayDataFn.create( defaultPntArray );
	    
	attributeAffects(achord0, outValue);
	attributeAffects(achord1, outValue);
	attributeAffects(achord2, outValue);
	attributeAffects(achord3, outValue);
	attributeAffects(athickness0, outValue);
	attributeAffects(athickness1, outValue);
	attributeAffects(athickness2, outValue);
	attributeAffects(athickness3, outValue);
	attributeAffects(anumfeather0, outValue);
	attributeAffects(anumfeather1, outValue);
	attributeAffects(anumfeather2, outValue);
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
	attributeAffects(adigitl, outValue);
	attributeAffects(ahumerusmat, outValue);
	attributeAffects(aulnamat, outValue);
	attributeAffects(ainboardmat, outValue);
	attributeAffects(amidsectmat0, outValue);
	attributeAffects(amidsectmat1, outValue);
	
	attributeAffects(aup0n0, outValue);
	attributeAffects(aup0n1, outValue);
	attributeAffects(aup0n2, outValue);
	attributeAffects(aup0n3, outValue);
	attributeAffects(aup0c0, outValue);
	attributeAffects(aup0c1, outValue);
	attributeAffects(aup0c2, outValue);
	attributeAffects(aup0c3, outValue);
	attributeAffects(aup0c4, outValue);
	attributeAffects(aup0t0, outValue);
	attributeAffects(aup0t1, outValue);
	attributeAffects(aup0t2, outValue);
	attributeAffects(aup0t3, outValue);
	attributeAffects(aup0t4, outValue);
	attributeAffects(aup0rz, outValue);
	
	attributeAffects(aup1n0, outValue);
	attributeAffects(aup1n1, outValue);
	attributeAffects(aup1n2, outValue);
	attributeAffects(aup1n3, outValue);
	attributeAffects(aup1c0, outValue);
	attributeAffects(aup1c1, outValue);
	attributeAffects(aup1c2, outValue);
	attributeAffects(aup1c3, outValue);
	attributeAffects(aup1c4, outValue);
	attributeAffects(aup1t0, outValue);
	attributeAffects(aup1t1, outValue);
	attributeAffects(aup1t2, outValue);
	attributeAffects(aup1t3, outValue);
	attributeAffects(aup1t4, outValue);
	attributeAffects(aup1rz, outValue);
	
	attributeAffects(alow0n0, outValue);
	attributeAffects(alow0n1, outValue);
	attributeAffects(alow0n2, outValue);
	attributeAffects(alow0n3, outValue);
	attributeAffects(alow0c0, outValue);
	attributeAffects(alow0c1, outValue);
	attributeAffects(alow0c2, outValue);
	attributeAffects(alow0c3, outValue);
	attributeAffects(alow0c4, outValue);
	attributeAffects(alow0t0, outValue);
	attributeAffects(alow0t1, outValue);
	attributeAffects(alow0t2, outValue);
	attributeAffects(alow0t3, outValue);
	attributeAffects(alow0t4, outValue);
	attributeAffects(alow0rz, outValue);
	
	attributeAffects(alow1n0, outValue);
	attributeAffects(alow1n1, outValue);
	attributeAffects(alow1n2, outValue);
	attributeAffects(alow1n3, outValue);
	attributeAffects(alow1c0, outValue);
	attributeAffects(alow1c1, outValue);
	attributeAffects(alow1c2, outValue);
	attributeAffects(alow1c3, outValue);
	attributeAffects(alow1c4, outValue);
	attributeAffects(alow1t0, outValue);
	attributeAffects(alow1t1, outValue);
	attributeAffects(alow1t2, outValue);
	attributeAffects(alow1t3, outValue);
	attributeAffects(alow1t4, outValue);
	attributeAffects(alow1rz, outValue);
	
	attributeAffects(ayawnoi, outValue);
	
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