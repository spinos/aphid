#include "ShrubVizNode.h"
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
#include <ExampVox.h>
#include <ExampData.h>
#include <mama/AttributeHelper.h>
#include <ogl/GlslInstancer.h>
#include <math/linearMath.h>

namespace aphid {

MTypeId ShrubVizNode::id( 0x7809778 );
MObject ShrubVizNode::aradiusMult;
MObject ShrubVizNode::ashrubbox;
MObject ShrubVizNode::ainsttrans;
MObject ShrubVizNode::ainstexamp;
MObject ShrubVizNode::ainexamp;
MObject ShrubVizNode::outValue;
	
ShrubVizNode::ShrubVizNode()
{ 
	memset(m_transBuf, 0, 64);
	m_cameraSpace = new Matrix44F;
	m_useExampleInput = true;
	attachSceneCallbacks(); 
}

ShrubVizNode::~ShrubVizNode() 
{	
	delete m_cameraSpace;
	detachSceneCallbacks(); 
}

MStatus ShrubVizNode::compute( const MPlug& plug, MDataBlock& block )
{
	if( plug == outValue ) {
	
		MDataHandle radiusMultH = block.inputValue(aradiusMult);
		float radiusScal = radiusMultH.asFloat();
		setGeomSizeMult(radiusScal);
	
		BoundingBox bb;
		getBBox(bb);
		setGeomBox(&bb);
	
		const int nexp = numExamples();
		AHelper::Info<int>("shrub viz n example", nexp);
		
		int nins = numInstances();
		
		if(nins < 1) {
			loadInternal();
			nins = numInstances();
		}
		AHelper::Info<int>("shrub viz n instance", nins);
		
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

void ShrubVizNode::draw( M3dView & view, const MDagPath & path, 
							 M3dView::DisplayStyle style,
							 M3dView::DisplayStatus status )
{
	MPlug radiusMultPlug(thisMObject(), aradiusMult);
	float radiusScal = radiusMultPlug.asFloat();
	setGeomSizeMult(radiusScal);
	
	BoundingBox bb;
	getBBox(bb);
	setGeomBox(&bb);
	
	const int nexp = numExamples();
//AHelper::Info<int>("nexp", nexp);
	
	if(nexp < 1) {
		return;
	}
	
	int nins = numInstances();
	
	if(nins < 1) {
		loadInternal();
		nins = numInstances();
	}
//AHelper::Info<int>("nins", nins);
	
	if(nins < 1) {
		return;
	}
	
	MDagPath cameraPath;
	view.getCamera(cameraPath);
	AHelper::GetViewMatrix(m_cameraSpace, cameraPath);
	Matrix33F mf = m_cameraSpace->rotation();
	mf *= geomSize();
    mf.glMatrix(m_transBuf);
		
	view.beginGL();
	glPointSize(2.f);
	drawBoundingBox(&bb);
	
	drawZCircle(m_transBuf);
	
	bool hasGlsl = isGlslReady();
	if(!hasGlsl ) {
		hasGlsl = prepareGlsl();
	}
	
	if(hasGlsl ) {

	drawWiredBoundInstances();
/// https://www.opengl.org/sdk/docs/man2/xhtml/glPushAttrib.xml	
	glPushAttrib(GL_ALL_ATTRIB_BITS);
		
	if ( style == M3dView::kFlatShaded || 
		    style == M3dView::kGouraudShaded ) {	

		drawSolidInstances();

	}
	else {
		drawWiredInstances();
	}
	
	glPopAttrib();
	
	} else {
		AHelper::Info<std::string >(" ERROR opengl ", "has no glsl");
	}
	
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
	
	aradiusMult = numFn.create( "radiusMultiplier", "rml", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(1.f);
	numFn.setMin(.05f);
	addAttribute(aradiusMult);
	
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
		MGlobal::displayWarning("failed add shrub box attrib");
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
	
	ainexamp = typFn.create("inExample", "ixmp", MFnData::kPlugin);
	typFn.setStorable(false);
	typFn.setConnectable(true);
	typFn.setArray(true);
	addAttribute(ainexamp);
		
    outValue = typFn.create( "outValue", "ov", MFnData::kPlugin );
	typFn.setStorable(false);
	typFn.setWritable(false);
	addAttribute(outValue);
	
	MPointArray defaultPntArray;
	MFnPointArrayData pntArrayDataFn;
	pntArrayDataFn.create( defaultPntArray );
	    
	attributeAffects(aradiusMult, outValue);
	attributeAffects(ainexamp, outValue);
	
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
	const int n = numInstances();
	AHelper::Info<int>(" n instance", n );
	MIntArray exmpi; exmpi.setLength(n);
	MVectorArray exmpt; exmpt.setLength(n<<2);
	
	for(int i=0;i<n;++i) {
		const InstanceD & ins = getInstance(i);
		exmpi[i] = ins._exampleId;
		int j = i<<2;
		const float * t = ins._trans;
		exmpt[j] = MVector(t[0], t[1], t[2]);
		exmpt[j+1] = MVector(t[4], t[5], t[6]);
		exmpt[j+2] = MVector(t[8], t[9], t[10]);
		exmpt[j+3] = MVector(t[12], t[13], t[14]);
	}
	
	MPlug exmpPlug(thisMObject(), ainstexamp);
	AttributeHelper::SaveArrayDataPlug<MIntArray, MFnIntArrayData>(exmpi, exmpPlug);
	
	MPlug transPlug(thisMObject(), ainsttrans);
	AttributeHelper::SaveArrayDataPlug<MVectorArray, MFnVectorArrayData>(exmpt, transPlug);
	
}

bool ShrubVizNode::loadInstances(const MVectorArray & instvecs,
						const MIntArray & instexmps)
{
	const int n = instexmps.length();
	if(n<1) {
		AHelper::Info<int>(" ShrubVizNode load no instance", n);
		return false;
	}
	
	if((n<<2) != instvecs.length() ) {
		AHelper::Info<unsigned>(" ShrubVizNode load wrong instance trans", instvecs.length() );
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
		addInstance(trans, instexmps[i]);
	}
	
	return true;
}

bool ShrubVizNode::loadInternal()
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

bool ShrubVizNode::loadInternal(MDataBlock& block)
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

MStatus ShrubVizNode::connectionMade ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	if(plug == ainexamp) {
		if(m_useExampleInput) addExample(plug);
	}
	return MPxLocatorNode::connectionMade (plug, otherPlug, asSrc );
}

MStatus ShrubVizNode::connectionBroken ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	if(plug.parent() == ainexamp) {
		AHelper::Info<MString>("disconnect", plug.name());
	}
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

void ShrubVizNode::addExample(const MPlug & plug)
{
	AHelper::Info<MString>(" ShrubVizNode add example", plug.name());
	MObject oslot;
	plug.getValue(oslot);
	MFnPluginData fslot(oslot);
	ExampData * dslot = (ExampData *)fslot.data();
	ExampVox * desc = dslot->getDesc();
	if(!desc) {
		AHelper::Info<MString>(" WARNING ShrubVizNode cannot get example data", plug.name());
		return;
	}
	addAExample(desc);
}

void ShrubVizNode::addInstance(const DenseMatrix<float> & trans,
					const int & exampleId)
{
	std::cout<<"\n add instance of exmp "<<exampleId;
	InstanceD ainstance;
	trans.extractData(ainstance._trans);
	ainstance._exampleId = exampleId;
	ainstance._instanceId = numInstances();
	addAInstance(ainstance);
}

void ShrubVizNode::drawWiredBoundInstances() const
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glEnableClientState(GL_VERTEX_ARRAY);
	
	const int nexp = numExamples();
	const int nins = numInstances();
	for(int i=0;i<nins;++i) {
		const InstanceD & ins = getInstance(i);
		glPushMatrix();
		glMultMatrixf(ins._trans);
		
		if(ins._exampleId < nexp) {
			//m_examples[ins._exampleId]->drawWiredBound();
			getExample(ins._exampleId)->drawAWireDop();
		} else {
			AHelper::Info<int>(" WARNING ShrubVizNode out of range example", ins._exampleId);
			AHelper::Info<int>(" instance", i);
		}
		
		glPopMatrix();
	}
	glDisableClientState(GL_VERTEX_ARRAY);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void ShrubVizNode::drawSolidInstances() const
{
	Vector3F lightVec(1,1,1);
	lightVec = m_cameraSpace->transformAsNormal(lightVec);
	m_instancer->setDistantLightVec(lightVec);
	
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);

	m_instancer->programBegin();
	
	const int nexp = numExamples();
	const int nins = numInstances();
	for(int i=0;i<nins;++i) {
		const InstanceD & ins = getInstance(i);
#if 0
		glPushMatrix();
		glMultMatrixf(ins._trans);
#endif
		const float *d = ins._trans;
	    glMultiTexCoord4f(GL_TEXTURE1, d[0], d[4], d[8], d[12]);
	    glMultiTexCoord4f(GL_TEXTURE2, d[1], d[5], d[9], d[13]);
	    glMultiTexCoord4f(GL_TEXTURE3, d[2], d[6], d[10], d[14]);
	    
		if(ins._exampleId < nexp) {
			const ExampVox * v = getExample(ins._exampleId);
			const float * c = v->diffuseMaterialColor();
			
			m_instancer->setDiffueColorVec(c);
			v->drawPoints();
		} else {
			AHelper::Info<int>(" WARNING ShrubVizNode out of range example", ins._exampleId);
			AHelper::Info<int>(" instance", i);
		}
		
#if 0
		glPopMatrix();
#endif
	}
	m_instancer->programEnd();
	
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}

void ShrubVizNode::drawWiredInstances() const
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glEnableClientState(GL_VERTEX_ARRAY);
	
	m_wireInstancer->programBegin();
	
	const int nexp = numExamples();
	const int nins = numInstances();
	
	for(int i=0;i<nins;++i) {
		const InstanceD & ins = getInstance(i);
		const float *d = ins._trans;
	    glMultiTexCoord4f(GL_TEXTURE1, d[0], d[4], d[8], d[12]);
	    glMultiTexCoord4f(GL_TEXTURE2, d[1], d[5], d[9], d[13]);
	    glMultiTexCoord4f(GL_TEXTURE3, d[2], d[6], d[10], d[14]);
	    
		if(ins._exampleId < nexp) {
			const ExampVox * v = getExample(ins._exampleId);
			const float * c = v->diffuseMaterialColor();
			
			m_wireInstancer->setDiffueColorVec(c);
			v->drawWiredPoints();
		} else {
			AHelper::Info<int>(" WARNING ShrubVizNode out of range example", ins._exampleId);
			AHelper::Info<int>(" instance", i);
		}
		
	}
	m_wireInstancer->programEnd();
	
	glDisableClientState(GL_VERTEX_ARRAY);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void ShrubVizNode::enableExampleInput()
{ m_useExampleInput = true; }
	
void ShrubVizNode::disableExampleInput()
{ m_useExampleInput = false; }

}
//:~