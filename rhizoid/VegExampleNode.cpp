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
#include <maya/MFnMeshData.h>
#include <maya/MFnPluginData.h>
#include <mama/AHelper.h>
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
MObject VegExampleNode::ainstexamp;
MObject VegExampleNode::ainsttrans;
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
	
		MDataHandle radiusMultH = block.inputValue(aradiusMult);
		float radiusScal = radiusMultH.asFloat();
		setGeomSizeMult(radiusScal);
	
		BoundingBox bb;
		getBBox(bb);
		setGeomBox(&bb);
	
		int nexp = numExamples();
		if(nexp < 1) {
			loadInternal();
			nexp = numExamples();
		}
		AHelper::Info<int>("vege viz n example", nexp);

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
	MPlug radiusMultPlug(thisMObject(), aradiusMult);
	float radiusScal = radiusMultPlug.asFloat();
	setGeomSizeMult(radiusScal);
	
	BoundingBox bb;
	getBBox(bb);
	setGeomBox(&bb);
	
	int nexp = numExamples();
//AHelper::Info<int>("nexp", nexp);
	
	if(nexp < 1) {
		loadInternal();
		nexp = numExamples();
	}
//AHelper::Info<int>("nins", nins);
	
	if(nexp < 1) {
		return;
	}
	
	MDagPath cameraPath;
	view.getCamera(cameraPath);
	AHelper::GetViewMatrix(m_cameraSpace, cameraPath);
	Matrix33F mf = m_cameraSpace->rotation();
	mf *= geomSize();
	
	float transF[16];
    mf.glMatrix(transF);
		
	view.beginGL();
	
	drawBoundingBox(&bb);
	drawZCircle(transF);

/// https://www.opengl.org/sdk/docs/man2/xhtml/glPushAttrib.xml	
	glPushAttrib(GL_ALL_ATTRIB_BITS | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
		
	glColor3f(1.f, 1.f, 1.f);
	glPointSize(2.f);
	
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
	numFn.setDefault(0.9f, 0.9f, 0.9f);
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
	numFn.setDefault(true);
	numFn.setMin(1);
	numFn.setMax(100);
	numFn.setDefault(1);
	addAttribute(avoxpriority);
	
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
											
	if(stat != MS::kSuccess) {
		MGlobal::displayWarning("failed create shrub box attrib");
	}
	
	typFn.setStorable(true);
	
	stat = addAttribute(ashrubbox);
	if(stat != MS::kSuccess) {
		MGlobal::displayWarning("failed add veg box attrib");
	}
	
	ainstbbox = typFn.create( "shrubExampleBox", "sbeb",
									MFnData::kVectorArray, vectArrayDataFn.object(),
									&stat );
											
	if(stat != MS::kSuccess) {
		MGlobal::displayWarning("failed create veg instance bbox attrib");
	}
	
	typFn.setStorable(true);
	
	stat = addAttribute(ainstbbox);
	if(stat != MS::kSuccess) {
		MGlobal::displayWarning("failed add veg instance bbox attrib");
	}
	
	ainstrange = typFn.create( "shrubIndRange", "sbir",
									MFnData::kIntArray, iArrayDataFn.object(),
									&stat );
	if(stat != MS::kSuccess) {
		MGlobal::displayWarning("failed create veg instance range attrib");
	}
	
	typFn.setStorable(true);
	
	stat = addAttribute(ainstrange);
	if(stat != MS::kSuccess) {
		MGlobal::displayWarning("failed add veg instance range attrib");
	}
	
	ainstexamp = typFn.create( "shrubExampleInd", "sbei",
									MFnData::kIntArray, iArrayDataFn.object(),
									&stat );
	if(stat != MS::kSuccess) {
		MGlobal::displayWarning("failed create shrub example ind attrib");
	}
	
	typFn.setStorable(true);
	
	stat = addAttribute(ainstexamp);
	if(stat != MS::kSuccess) {
		MGlobal::displayWarning("failed add shrub example ind attrib");
	}
	
	ainsttrans = typFn.create( "shrubExampleTrans", "sbet",
									MFnData::kVectorArray, vectArrayDataFn.object(),
									&stat );
	if(stat != MS::kSuccess) {
		MGlobal::displayWarning("failed create shrub example trans attrib");
	}
	
	typFn.setStorable(true);
	
	stat = addAttribute(ainsttrans);
	if(stat != MS::kSuccess) {
		MGlobal::displayWarning("failed add shrub example trans attrib");
	}
		
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
	    
	attributeAffects(aradiusMult, outValue);
	attributeAffects(adrawColorR, outValue);
	attributeAffects(adrawColorG, outValue);
	attributeAffects(adrawColorB, outValue);
	attributeAffects(adrawDopSizeX, outValue);
	attributeAffects(adrawDopSizeY, outValue);
	attributeAffects(adrawDopSizeZ, outValue);
	attributeAffects(avoxactive, outValue);
	attributeAffects(avoxvisible, outValue);
	
	return MS::kSuccess;
}

bool VegExampleNode::loadInstances(const MVectorArray & instvecs,
						const MIntArray & instexmps)
{
	const int n = instexmps.length();
	if(n<1) {
		AHelper::Info<int>(" VegExampleNode load no instance", n);
		return false;
	}
	
	if((n<<2) != instvecs.length() ) {
		AHelper::Info<unsigned>(" VegExampleNode load wrong instance trans", instvecs.length() );
		return false;
	}
	
	DenseMatrix<float> trans(4,4);
	trans.setIdentity();
	
	for(int i=0;i<n;++i) {
		for(int j =0;j<4;++j) {
			const MVector & a = instvecs[(i<<2) + j];
			float * cj = trans.column(j);
			cj[0] = a.x;
			cj[1] = a.y;
			cj[2] = a.z;
		}
		
	}
	
	return true;
}

bool VegExampleNode::loadInternal()
{
	AHelper::Info<MString>("shrub load internal", MFnDependencyNode(thisMObject()).name() );
	
	MPlug exmpPlug(thisMObject(), ainstexamp);
	MIntArray insti;
	AttributeHelper::LoadArrayDataPlug<MIntArray, MFnIntArrayData>(insti, exmpPlug);
	
	MPlug transPlug(thisMObject(), ainsttrans);
	MVectorArray instt;
	AttributeHelper::LoadArrayDataPlug<MVectorArray, MFnVectorArrayData>(instt, transPlug);
	
	return loadInstances(instt, insti);
}

bool VegExampleNode::loadInternal(MDataBlock& block)
{
	AHelper::Info<MString>("shrub load internal", MFnDependencyNode(thisMObject()).name() );
	
	MDataHandle exmpH = block.inputValue(ainstexamp);
	MIntArray insti;
	AttributeHelper::LoadArrayDataHandle<MIntArray, MFnIntArrayData>(insti, exmpH);
	
	MDataHandle transH = block.inputValue(ainsttrans);
	MVectorArray instt;
	AttributeHelper::LoadArrayDataHandle<MVectorArray, MFnVectorArrayData>(instt, transH);
	
	return loadInstances(instt, insti);
}

void VegExampleNode::saveBBox(const BoundingBox & bbox)
{
	MVectorArray dbox; dbox.setLength(2);
	dbox[0] = MVector(bbox.getMin(0), bbox.getMin(1), bbox.getMin(2) );
	dbox[1] = MVector(bbox.getMax(0), bbox.getMax(1), bbox.getMax(2) );
	MFnVectorArrayData vecFn;
	MObject obox = vecFn.create(dbox);
	MPlug dboxPlug(thisMObject(), ashrubbox );
	dboxPlug.setValue(obox);
}

void VegExampleNode::getBBox(BoundingBox & bbox) const
{
	MPlug dboxPlug(thisMObject(), ashrubbox);
	MObject obox;
	dboxPlug.getValue(obox);
	
	MFnVectorArrayData vecFn(obox);
	MVectorArray dbox = vecFn.array();
	
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
	saveGroupBBox();
	const int totalNInst = saveGroupRange();
	AHelper::Info<unsigned>(" n inst", totalNInst );
}

void VegExampleNode::saveGroupBBox()
{
	const int nexmp = numExamples();
	MVectorArray dbox; dbox.setLength(nexmp);
	for(int i=0;i<nexmp;++i) {
		CompoundExamp * cxmp = getCompoundExample(i);
		const BoundingBox & gbx = cxmp->geomBox();
		dbox[i*2] = MVector(gbx.getMin(0), gbx.getMin(1), gbx.getMin(2) );
		dbox[i*2 + 1] = MVector(gbx.getMax(0), gbx.getMax(1), gbx.getMax(2) );
	}
	MFnVectorArrayData vecFn;
	MObject obox = vecFn.create(dbox);
	MPlug dboxPlug(thisMObject(), ainstbbox );
	dboxPlug.setValue(obox);
}

int VegExampleNode::saveGroupRange()
{
	const int nexmp = numExamples();
	MIntArray drange; drange.setLength(nexmp+1);
	int b = 0;
	for(int i=0;i<nexmp;++i) {
		CompoundExamp * cxmp = getCompoundExample(i);
		drange[i] = b;
		const int c = cxmp->numInstances();
		b += c;
	}
	drange[nexmp] = b;
	MFnIntArrayData vecFn;
	MObject orange = vecFn.create(drange);
	MPlug drangePlug(thisMObject(), ainstrange );
	drangePlug.setValue(orange);
	return b;
}

}
//:~