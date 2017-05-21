#include "VegExampleNode.h"
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnVectorArrayData.h>
#include <maya/MFnMatrixData.h>
#include <maya/MFnPointArrayData.h>
#include <maya/MFnDoubleArrayData.h>
#include <maya/MFnIntArrayData.h>
#include <maya/MPointArray.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MFnPluginData.h>
#include <mama/AHelper.h>
#include <mama/AttributeHelper.h>
#include <CompoundExamp.h>
#include <ExampData.h>
#include <mama/AttributeHelper.h>
#include <ogl/GlslInstancer.h>
#include <math/linearMath.h>

namespace aphid {

MTypeId VegExampleNode::id( 0xdb02444 );
MObject VegExampleNode::ashrubbox;
MObject VegExampleNode::ainstbbox;
MObject VegExampleNode::ainstrange;
MObject VegExampleNode::ainstinds;
MObject VegExampleNode::ainsttrans;
MObject VegExampleNode::apntPosNmlCol;
MObject VegExampleNode::apntRange;
MObject VegExampleNode::ahullPosNml;
MObject VegExampleNode::ahullRange;
MObject VegExampleNode::adrawColor;
MObject VegExampleNode::adrawColorR;
MObject VegExampleNode::adrawColorG;
MObject VegExampleNode::adrawColorB;
MObject VegExampleNode::adrawDopSizeX;
MObject VegExampleNode::adrawDopSizeY;
MObject VegExampleNode::adrawDopSizeZ;
MObject VegExampleNode::adrawDopSize;
MObject VegExampleNode::aradiusMult;
MObject VegExampleNode::aininstspace;
MObject VegExampleNode::avoxactive;
MObject VegExampleNode::avoxvisible;
MObject VegExampleNode::avoxpriority;
MObject VegExampleNode::adrawVoxTag;
MObject VegExampleNode::avarp;
MObject VegExampleNode::outValue;
	
VegExampleNode::VegExampleNode()
{ 
	m_cameraSpace = new Matrix44F;
}

VegExampleNode::~VegExampleNode() 
{	
	delete m_cameraSpace;
}

MStatus VegExampleNode::compute( const MPlug& plug, MDataBlock& block )
{
	if( plug == outValue ) {
	
		MDataHandle rch = block.inputValue(adrawColorR);
		MDataHandle gch = block.inputValue(adrawColorG);
		MDataHandle bch = block.inputValue(adrawColorB);
		setDiffuseMaterialCol3(rch.asFloat(), gch.asFloat(), bch.asFloat() );
	
		MDataHandle szxh = block.inputValue(adrawDopSizeX);
		MDataHandle szyh = block.inputValue(adrawDopSizeY);
		MDataHandle szzh = block.inputValue(adrawDopSizeZ);
		setDopSize(szxh.asFloat(), szyh.asFloat(), szzh.asFloat() );
	
		MDataHandle radiusMultH = block.inputValue(aradiusMult);
		float radiusScal = radiusMultH.asFloat();
		setGeomSizeMult(radiusScal);
	
		MDataHandle detailTypeHandle = block.inputValue( adrawVoxTag );
		const short detailType = detailTypeHandle.asShort();
		setDetailDrawType(detailType);

		int nexp = numExamples();
		if(nexp < 1) {
			loadInternal();
			nexp = numExamples();
		}
		AHelper::Info<int>("vege viz n example", nexp);
		
		updateAllGeomSize();
		updateAllDop();
		updateAllDetailDrawType();
		
		const bool vis = block.inputValue(avoxvisible).asBool(); 
		setVisible(vis);
		
		MFnPluginData fnPluginData;
		MStatus status;
		MObject newDataObject = fnPluginData.create(ExampData::id, &status);
		
		ExampData * pData = (ExampData *) fnPluginData.data(&status);
		
		if(pData) pData->setDesc(this);
		
		MDataHandle outputHandle = block.outputValue( outValue );
		outputHandle.set( pData );
		block.setClean(plug);
    }

	return MS::kSuccess;
}

void VegExampleNode::draw( M3dView & view, const MDagPath & path, 
							 M3dView::DisplayStyle style,
							 M3dView::DisplayStatus status )
{
	MObject selfNode = thisMObject();
	
	MPlug rPlug(selfNode, adrawColorR);
	MPlug gPlug(selfNode, adrawColorG);
	MPlug bPlug(selfNode, adrawColorB);
	setDiffuseMaterialCol3(rPlug.asFloat(), gPlug.asFloat(), bPlug.asFloat() );
	
	MPlug szxp(selfNode, adrawDopSizeX);
	MPlug szyp(selfNode, adrawDopSizeY);
	MPlug szzp(selfNode, adrawDopSizeZ);
	setDopSize(szxp.asFloat(), szyp.asFloat(), szzp.asFloat() );
	
	MPlug radiusMultPlug(thisMObject(), aradiusMult);
	float radiusScal = radiusMultPlug.asFloat();
	setGeomSizeMult(radiusScal);
	
	int nexp = numExamples();

	if(nexp < 1) {
		loadInternal();
		nexp = numExamples();
	}
	
	if(nexp < 1) {
		AHelper::Info<int>("vege viz has no data", nexp);
		return;
	}
	
	updateAllGeomSize();
	updateAllDop();
	
	MDagPath cameraPath;
	view.getCamera(cameraPath);
	AHelper::GetViewMatrix(m_cameraSpace, cameraPath);
	Matrix33F mf = m_cameraSpace->rotation();
	mf *= geomSize();
	
	float transF[16];
    mf.glMatrix(transF);
		
	view.beginGL();
	
	const BoundingBox & bb = geomBox();
	
	drawBoundingBox(&bb);
	drawZCircle(transF);

/// https://www.opengl.org/sdk/docs/man2/xhtml/glPushAttrib.xml	
	glPushAttrib(GL_ALL_ATTRIB_BITS | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
		
	glColor3f(1.f, 1.f, 1.f);
	glPointSize(2.f);
	
	drawExampPoints(0);
	drawExampHull(0);
	
	if ( style == M3dView::kFlatShaded || 
		    style == M3dView::kGouraudShaded ) {	

	} else {
	
	}
	
	glPopAttrib();
	
	view.endGL();
}

bool VegExampleNode::isBounded() const
{ return true; }

MBoundingBox VegExampleNode::boundingBox() const
{   
	BoundingBox bbox(-1,-1,-1,1,1,1);
	
	getBBox(bbox);
	
	MPoint corner1(bbox.getMin(0), bbox.getMin(1), bbox.getMin(2));
	MPoint corner2(bbox.getMax(0), bbox.getMax(1), bbox.getMax(2));

	return MBoundingBox( corner1, corner2 );
}

void* VegExampleNode::creator()
{
	return new VegExampleNode();
}

MStatus VegExampleNode::initialize()
{ 
	MFnNumericAttribute numFn;
	MFnTypedAttribute typFn;
	MFnEnumAttribute	enumAttr;
	MStatus			 stat;
	
	adrawColorR = numFn.create( "dspColorR", "dspr", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setDefault(0.47f);
	numFn.setMin(0.f);
	numFn.setMax(1.f);
	addAttribute(adrawColorR);
	
	adrawColorG = numFn.create( "dspColorG", "dspg", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setDefault(0.46f);
	numFn.setMin(0.f);
	numFn.setMax(1.f);
	addAttribute(adrawColorG);
	
	adrawColorB = numFn.create( "dspColorB", "dspb", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setDefault(0.45f);
	numFn.setMin(0.f);
	numFn.setMax(1.f);
	addAttribute(adrawColorB);
	
	adrawColor = numFn.create( "dspColor", "dspc", adrawColorR, adrawColorG, adrawColorB );
	numFn.setStorable(true);
	numFn.setUsedAsColor(true);
	numFn.setDefault(0.47f, 0.46f, 0.45f);
	addAttribute(adrawColor);
	
	adrawDopSizeX = numFn.create( "dspDopX", "ddpx", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.9f);
	numFn.setMin(0.1f);
	numFn.setMax(1.f);
	addAttribute(adrawDopSizeX);
	
	adrawDopSizeY = numFn.create( "dspDopY", "ddpy", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.9f);
	numFn.setMin(0.1f);
	numFn.setMax(1.f);
	addAttribute(adrawDopSizeY);
	
	adrawDopSizeZ = numFn.create( "dspDopZ", "ddpz", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.9f);
	numFn.setMin(0.1f);
	numFn.setMax(1.f);
	addAttribute(adrawDopSizeZ);
	
	adrawDopSize = numFn.create( "dspDop", "ddps", adrawDopSizeX, adrawDopSizeY, adrawDopSizeZ );
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setUsedAsColor(true);
	numFn.setDefault(0.9f, 1.f, 0.9f);
	addAttribute(adrawDopSize);
	
	aradiusMult = numFn.create( "radiusMultiplier", "rml", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(1.f);
	numFn.setMin(.05f);
	addAttribute(aradiusMult);
	
	avoxactive = numFn.create( "exampleActive", "exa", MFnNumericData::kBoolean);
	numFn.setStorable(true);
	numFn.setDefault(true);
	addAttribute(avoxactive);
	
	avoxvisible = numFn.create( "exampleVisible", "exv", MFnNumericData::kBoolean);
	numFn.setStorable(true);
	numFn.setDefault(true);
	addAttribute(avoxvisible);
	
	avoxpriority = numFn.create( "examplePriority", "expi", MFnNumericData::kShort);
	numFn.setStorable(true);
	numFn.setMin(1);
	numFn.setMax(100);
	numFn.setDefault(1);
	addAttribute(avoxpriority);
	
/// synth pattern
	avarp = numFn.create( "variablePattern", "varp", MFnNumericData::kShort);
	numFn.setStorable(true);
	numFn.setMin(0);
	numFn.setMax(100);
	numFn.setDefault(0);
	addAttribute(avarp);
	
	MDoubleArray defaultDArray;
	MFnDoubleArrayData dArrayDataFn;
	dArrayDataFn.create( defaultDArray );
	
	MVectorArray defaultVectArray;
	MFnVectorArrayData vectArrayDataFn;
	vectArrayDataFn.create( defaultVectArray );
	
	MIntArray defaultIArray;
	MFnIntArrayData iArrayDataFn;
	iArrayDataFn.create( defaultIArray );
	
	ashrubbox = typFn.create( "shrubBox", "sbbx",
									MFnData::kVectorArray, vectArrayDataFn.object(),
									&stat );
	AHelper::CheckWarningStat(stat, "failed create shrub box attrib");
	
	typFn.setStorable(true);
	
	stat = addAttribute(ashrubbox);
	AHelper::CheckWarningStat(stat, "failed add veg box attrib");
	
	ainstbbox = typFn.create( "shrubExampleBox", "sbeb",
									MFnData::kVectorArray, vectArrayDataFn.object(),
									&stat );
	AHelper::CheckWarningStat(stat, "failed create veg instance bbox attrib");
	
	typFn.setStorable(true);
	
	stat = addAttribute(ainstbbox);
	AHelper::CheckWarningStat(stat, "failed add veg instance bbox attrib");
	
	ainstrange = typFn.create( "shrubIndRange", "sbir",
									MFnData::kIntArray, iArrayDataFn.object(),
									&stat );
	AHelper::CheckWarningStat(stat, "failed create veg instance range attrib");
	
	typFn.setStorable(true);
	
	stat = addAttribute(ainstrange);
	AHelper::CheckWarningStat(stat, "failed add veg instance range attrib");
	
/// instance ind and tm
	ainstinds = typFn.create( "shrubExampleInd", "sbei",
									MFnData::kIntArray, iArrayDataFn.object(),
									&stat );
	AHelper::CheckWarningStat(stat, "failed create veg example ind attrib");
	
	typFn.setStorable(true);
	
	stat = addAttribute(ainstinds);
	AHelper::CheckWarningStat(stat, "failed add veg example ind attrib");
	
	ainsttrans = typFn.create( "shrubExampleTrans", "sbet",
									MFnData::kVectorArray, vectArrayDataFn.object(),
									&stat );
	AHelper::CheckWarningStat(stat, "failed create veg example trans attrib");
	
	typFn.setStorable(true);
	
	stat = addAttribute(ainsttrans);
	AHelper::CheckWarningStat(stat, "failed add shrub example trans attrib");
	
/// points	
	apntPosNmlCol = typFn.create( "pntPNC", "ppnc",
									MFnData::kVectorArray, vectArrayDataFn.object(),
									&stat );
	AHelper::CheckWarningStat(stat, "failed create ppnc attrib");
	
	typFn.setStorable(true);
	
	stat = addAttribute(apntPosNmlCol);
	AHelper::CheckWarningStat(stat, "failed add ppnc attrib");
	
	apntRange = typFn.create( "pntRange", "prng",
									MFnData::kIntArray, iArrayDataFn.object(),
									&stat );
	AHelper::CheckWarningStat(stat, "failed create veg pnt range attrib");
	
	typFn.setStorable(true);
	
	stat = addAttribute(apntRange);
	AHelper::CheckWarningStat(stat, "failed add veg pnt range attrib");
	
/// hull
	ahullPosNml = typFn.create( "hullPNC", "hpnc",
									MFnData::kVectorArray, vectArrayDataFn.object(),
									&stat );
											
	AHelper::CheckWarningStat(stat, "failed create hull pn attrib");
	
	typFn.setStorable(true);
	
	stat = addAttribute(ahullPosNml);
	AHelper::CheckWarningStat(stat, "failed add hull pn attrib");
	
	ahullRange = typFn.create( "hullRange", "hlng",
									MFnData::kIntArray, iArrayDataFn.object(),
									&stat );
	AHelper::CheckWarningStat(stat, "failed create veg hull range attrib");
	
	typFn.setStorable(true);
	
	stat = addAttribute(ahullRange);
	AHelper::CheckWarningStat(stat, "failed add veg hull range attrib");
	
    outValue = typFn.create( "outValue", "ov", MFnData::kPlugin );
	typFn.setStorable(false);
	typFn.setWritable(false);
	addAttribute(outValue);
	
	MPointArray defaultPntArray;
	MFnPointArrayData pntArrayDataFn;
	pntArrayDataFn.create( defaultPntArray );
	
	MFnMatrixAttribute matAttr;
	aininstspace = matAttr.create("instanceSpace", "sinst", MFnMatrixAttribute::kDouble);
	matAttr.setStorable(false);
	matAttr.setWritable(true);
	matAttr.setConnectable(true);
    matAttr.setArray(true);
    matAttr.setDisconnectBehavior(MFnAttribute::kDelete);
	addAttribute( aininstspace );
	
	adrawVoxTag = enumAttr.create( "dspDetailType", "ddt", 0, &stat );
	enumAttr.addField( "point", 0 );
	enumAttr.addField( "grid", 1 );
	enumAttr.setHidden( false );
	enumAttr.setKeyable( true );
	addAttribute(adrawVoxTag);
	    
	attributeAffects(aradiusMult, outValue);
	attributeAffects(adrawColorR, outValue);
	attributeAffects(adrawColorG, outValue);
	attributeAffects(adrawColorB, outValue);
	attributeAffects(adrawDopSizeX, outValue);
	attributeAffects(adrawDopSizeY, outValue);
	attributeAffects(adrawDopSizeZ, outValue);
	attributeAffects(avoxactive, outValue);
	attributeAffects(avoxvisible, outValue);
	attributeAffects(adrawVoxTag, outValue);
	return MS::kSuccess;
}

bool VegExampleNode::loadInternal()
{
	MObject onode = thisMObject();
	loadBBox();
	
	MPlug dboxPlug(onode, ainstbbox );
	loadGroupBBox(dboxPlug);
	
	short pat = 0;
	MPlug patPlug(onode, avarp );
	pat = patPlug.asShort();
	std::cout<<"\n synth pattern "<<pat;
	setShortPattern(pat);
	
	MPlug drangePlug(onode, ainstrange );
	MPlug dindPlug(onode, ainstinds );
	MPlug dtmPlug(onode, ainsttrans );
	loadInstance(drangePlug, dindPlug, dtmPlug);
	
	MPlug dpntPncPlug(onode, apntPosNmlCol );
	MPlug dpntRangePlug(onode, apntRange );
	loadPoints(dpntRangePlug, dpntPncPlug);
	
	MPlug dhullPnPlug(onode, ahullPosNml );
	MPlug dhullRangePlug(onode, ahullRange );
	loadHull(dhullRangePlug, dhullPnPlug );
	
	buildAllExmpVoxel();
	std::cout.flush();
	
	return true;
}

void VegExampleNode::loadBBox()
{
	BoundingBox dbox;
	getBBox(dbox);
	setGeomBox(&dbox);
}

void VegExampleNode::saveBBox(const BoundingBox & bbox)
{
	MVectorArray dbox; dbox.setLength(2);
	dbox[0] = MVector(bbox.getMin(0), bbox.getMin(1), bbox.getMin(2) );
	dbox[1] = MVector(bbox.getMax(0), bbox.getMax(1), bbox.getMax(2) );
	
	MPlug dboxPlug(thisMObject(), ashrubbox );
	AttributeHelper::SaveArrayDataPlug<MVectorArray, MFnVectorArrayData > (dbox, dboxPlug);
}

void VegExampleNode::getBBox(BoundingBox & bbox) const
{
	MVectorArray dbox;
	MPlug dboxPlug(thisMObject(), ashrubbox);
	
	AttributeHelper::LoadArrayDataPlug<MVectorArray, MFnVectorArrayData > (dbox, dboxPlug);
	
	if(dbox.length() < 2) {
		AHelper::Info<unsigned>(" WARNING VegExampleNode getBBox invalid data n", dbox.length() );
		bbox.setMin(-1.f, -1.f, -1.f);
		bbox.setMax(1.f, 1.f, 1.f);
		return;
	}
	
	bbox.setMin(dbox[0].x, dbox[0].y, dbox[0].z );
	bbox.setMax(dbox[1].x, dbox[1].y, dbox[1].z );
}

void VegExampleNode::saveInternal()
{
	const int nexmp = numExamples();
	AHelper::Info<int>(" VegExampleNode::saveInternal n exmp", nexmp );
	if(nexmp < 1) {
		return;
	}
	saveBBox(geomBox() );
	
	MObject onode = thisMObject();
	
	short pat = getPattern();
	MPlug patPlug(onode, avarp );
	patPlug.setValue(pat);
	
	MPlug dboxPlug(onode, ainstbbox );
	saveGroupBBox(dboxPlug);
	
	MPlug drangePlug(onode, ainstrange );
	MPlug dindPlug(onode, ainstinds );
	MPlug dtmPlug(onode, ainsttrans );
	saveInstance(drangePlug, dindPlug, dtmPlug);
	
	MPlug dpntPncPlug(onode, apntPosNmlCol );
	MPlug dpntRangePlug(onode, apntRange );
	savePoints(dpntRangePlug, dpntPncPlug);
	
	MPlug dhullPnPlug(onode, ahullPosNml );
	MPlug dhullRangePlug(onode, ahullRange );
	saveHull(dhullRangePlug, dhullPnPlug);
	
	std::cout.flush();
}
	
}
//:~