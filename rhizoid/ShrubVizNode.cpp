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
#include <maya/MFnPluginData.h>
#include <AHelper.h>
#include <ExampVox.h>
#include <ExampData.h>
#include <math/linearMath.h>

namespace aphid {

MTypeId ShrubVizNode::id( 0x7809778 );
MObject ShrubVizNode::ashrubbox;
MObject ShrubVizNode::ainexamp;
MObject ShrubVizNode::outValue;

ShrubVizNode::ShrubVizNode()
{ attachSceneCallbacks(); }

ShrubVizNode::~ShrubVizNode() 
{ 
	m_instances.clear();
	m_examples.clear();
	detachSceneCallbacks(); 
}

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
	const int nexp = numExamples();
	
	if(nexp < 1) {
		return;
	}
	
	const int nins = numInstances();
	
	if(nins < 1) {
		return;
	}
	
	MObject thisNode = thisMObject();
		
	view.beginGL();
	
	glPushMatrix();
	
	BoundingBox bbox;
	getBBox(bbox);
	
	drawBoundingBox(&bbox);
	
	drawWiredBoundInstances();

	if ( style == M3dView::kFlatShaded || 
		    style == M3dView::kGouraudShaded ) {	
			
		glDepthFunc(GL_LEQUAL);
		glPushAttrib(GL_LIGHTING_BIT);
		glEnable(GL_LIGHTING);
			
		drawSolidInstances();
		
		glDisable(GL_LIGHTING);
		glPopAttrib();
	}
	else {
		drawWiredInstances();
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
	
	ainexamp = typFn.create("inExample", "ixmp", MFnData::kPlugin);
	typFn.setStorable(false);
	typFn.setConnectable(true);
	typFn.setArray(true);
	addAttribute(ainexamp);
		
    outValue = numFn.create( "outValue", "ov", MFnNumericData::kFloat );
	numFn.setStorable(false);
	numFn.setWritable(false);
	addAttribute(outValue);
	
	MPointArray defaultPntArray;
	MFnPointArrayData pntArrayDataFn;
	pntArrayDataFn.create( defaultPntArray );
	    
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
}

bool ShrubVizNode::loadInternal(MDataBlock& block)
{
	AHelper::Info<MString>("shrub load internal", MFnDependencyNode(thisMObject()).name() );
	return true;
}

MStatus ShrubVizNode::connectionMade ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	if(plug == ainexamp) {
		addExample(plug);
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

int ShrubVizNode::numInstances() const
{ return m_instances.size(); }

int ShrubVizNode::numExamples() const
{ return m_examples.size(); }

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
	m_examples.push_back(desc);
}

void ShrubVizNode::addInstance(const DenseMatrix<float> & trans,
					const int & exampleId)
{
	std::cout<<"\n add instance of exmp "<<exampleId;
	InstanceD ainstance;
	trans.extractData(ainstance._trans);
	ainstance._exampleId = exampleId;
	ainstance._instanceId = m_instances.size();
	m_instances.push_back(ainstance);
}

void ShrubVizNode::drawWiredBoundInstances() const
{
	const int nexp = numExamples();
	const int nins = numInstances();
	for(int i=0;i<nins;++i) {
		const InstanceD & ins = m_instances[i];
		glPushMatrix();
		glMultMatrixf(ins._trans);
		
		if(ins._exampleId < nexp) {
			m_examples[ins._exampleId]->drawWiredBound();
		} else {
			AHelper::Info<int>(" WARNING ShrubVizNode out of range example", ins._exampleId);
			AHelper::Info<int>(" instance", i);
		}
		
		glPopMatrix();
	}
}

void ShrubVizNode::drawSolidInstances() const
{
	const int nexp = numExamples();
	const int nins = numInstances();
	for(int i=0;i<nins;++i) {
		const InstanceD & ins = m_instances[i];
		glPushMatrix();
		glMultMatrixf(ins._trans);
		
		if(ins._exampleId < nexp) {
			m_examples[ins._exampleId]->drawSolidTriangles();
		} else {
			AHelper::Info<int>(" WARNING ShrubVizNode out of range example", ins._exampleId);
			AHelper::Info<int>(" instance", i);
		}
		
		glPopMatrix();
	}
}

void ShrubVizNode::drawWiredInstances() const
{
	const int nexp = numExamples();
	const int nins = numInstances();
	for(int i=0;i<nins;++i) {
		const InstanceD & ins = m_instances[i];
		glPushMatrix();
		glMultMatrixf(ins._trans);
		
		if(ins._exampleId < nexp) {
			m_examples[ins._exampleId]->drawWiredTriangles();
		} else {
			AHelper::Info<int>(" WARNING ShrubVizNode out of range example", ins._exampleId);
			AHelper::Info<int>(" instance", i);
		}
		
		glPopMatrix();
	}
}

}
//:~