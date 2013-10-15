#include "MaraNode.h"
#include <maya/MString.h> 
#include <maya/MGlobal.h>

#include <maya/MVector.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MDataHandle.h>
#include <maya/MColor.h>
#include <maya/MDistance.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MVectorArray.h>
#include <maya/MFnVectorArrayData.h>
#include <maya/MEulerRotation.h>
#include <maya/MFnMatrixData.h>
#include <maya/MFnDoubleArrayData.h>
#include <maya/MIntArray.h>
#include <maya/MPointArray.h>
#include <maya/MFnMeshData.h>
#include <maya/MItMeshPolygon.h>
#include <MlDrawer.h>
#include <MlScene.h>
#include <MlSkin.h>

MTypeId MallardViz::id( 0x52e8a1 );
MObject MallardViz::outValue;
MObject MallardViz::aframe;
MObject MallardViz::acachename;
MObject MallardViz::ainmesh;

MallardViz::MallardViz() : fHasView(0) 
{
    m_cache = new MlDrawer;
    m_scene = new MlScene;
}

MallardViz::~MallardViz() 
{
	if(m_scene->isOpened()) m_scene->close();
	delete m_scene;
    if(m_cache->isOpened()) m_cache->close();
    delete m_cache;
}

MStatus MallardViz::compute( const MPlug& plug, MDataBlock& block )
{
	if( plug == outValue ) {
		
		updateWorldSpace();
		
		MStatus status;
		
		double dtime = block.inputValue( aframe ).asDouble();
		int iframe = int(float(int(dtime * 1000 + 0.5))/1000.f);
		m_bodyMesh = block.inputValue( ainmesh ).asMesh();
		MString filename =  block.inputValue( acachename ).asString();	
		
		loadCache(filename.asChar());
		
		m_cache->setCurrentFrame(iframe);
		if(m_scene->isOpened()) m_cache->readBuffer(m_scene->skin());
		float result = 1.f;

		MDataHandle outputHandle = block.outputValue( outValue );
		outputHandle.set( result );
		block.setClean(plug);
    }

	return MS::kSuccess;
}

void MallardViz::draw( M3dView & view, const MDagPath & path, 
							 M3dView::DisplayStyle style,
							 M3dView::DisplayStatus status )
{ 	
	updateWorldSpace();
	MObject thisNode = thisMObject();
	
	_viewport = view;
	fHasView = 1;

	view.beginGL();
	
	double mm[16];
	char displayMeshIsValid = hasDisplayMesh();
	
	const GLfloat grayDiffuseMaterial[] = {0.47, 0.46, 0.45};
	const GLfloat greenDiffuseMaterial[] = {0.33, 0.53, 0.37}; 
	
	//if ( ( style == M3dView::kFlatShaded ) || 
	//	    ( style == M3dView::kGouraudShaded ) ) {
		glPushAttrib(GL_LIGHTING_BIT);
		glEnable(GL_LIGHTING);
		glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, grayDiffuseMaterial);
		
            glPushMatrix();
    
			if(m_scene->isOpened()) m_cache->draw(m_scene->skin());	
            glPopMatrix();
        
		glDisable(GL_LIGHTING);
		glPopAttrib();
	//}
	
	view.endGL();
}

bool MallardViz::isBounded() const
{ 
	return true;
}

MBoundingBox MallardViz::boundingBox() const
{   
	
	MPoint corner1(-1, -1, -1);
	MPoint corner2(1, 1, 1);

	return MBoundingBox( corner1, corner2 );
}

void* MallardViz::creator()
{
	return new MallardViz();
}

MStatus MallardViz::initialize()
{ 
	MFnNumericAttribute numFn;
	MStatus			 stat;

	MFnTypedAttribute typedAttrFn;
	MVectorArray defaultVectArray;
	MFnVectorArrayData vectArrayDataFn;
	vectArrayDataFn.create( defaultVectArray );	

    aframe = numFn.create( "currentTime", "ct", MFnNumericData::kDouble, 1.0 );
	numFn.setStorable(true);
	numFn.setKeyable(true);
	addAttribute( aframe );
	    
    outValue = numFn.create( "outValue", "ov", MFnNumericData::kFloat );
	numFn.setStorable(false);
	numFn.setWritable(false);
	addAttribute(outValue);
	
	MFnTypedAttribute   stringAttr;
	acachename = stringAttr.create( "cachePath", "cp", MFnData::kString );
 	stringAttr.setStorable(true);
	addAttribute( acachename );
	
	ainmesh = typedAttrFn.create("displayMesh", "dspm", MFnMeshData::kMesh);
	typedAttrFn.setStorable(false);
	typedAttrFn.setWritable(true);
	typedAttrFn.setConnectable(true);
	addAttribute( ainmesh );
	
	attributeAffects(ainmesh, outValue);
	attributeAffects(acachename, outValue);
	attributeAffects(aframe, outValue);
	return MS::kSuccess;
}

void MallardViz::loadCache(const char* filename)
{
	if(std::string(filename) == m_cache->fileName()) return;

    MGlobal::displayInfo("Mallard viz loading...");
	if(!m_cache->open(filename)) {
		MGlobal::displayWarning(MString("Mallard viz cannot read cache from ") + filename);
		return;
	}
	MGlobal::displayInfo(MString("Mallard viz read cache from ") + filename);
	std::string sceneName = m_cache->readSceneName();
	if(sceneName != "unknown") loadScene(sceneName.c_str());
	    
	if(m_scene->isOpened()) {
		const unsigned nc = m_scene->skin()->numFeathers();
		MlCalamus::FeatherLibrary = m_scene;
		m_cache->computeBufferIndirection(m_scene->skin());
	}
}

void MallardViz::loadScene(const char* filename)
{
    if(filename == m_scene->fileName()) return;
        
    m_scene->open(filename);
	MGlobal::displayInfo(MString("scene is opened ")+ m_scene->fileName().c_str());
	
}

void MallardViz::setCullMesh(MDagPath mesh)
{
	MGlobal::displayInfo(MString("Mallard viz uses blocker: ") + mesh.fullPathName());
	
}

void MallardViz::updateWorldSpace()
{
	MObject thisNode = thisMObject();
	MDagPath thisPath;
	MDagPath::getAPathTo(thisNode, thisPath);
	_worldSpace = thisPath.inclusiveMatrix();
	_worldInverseSpace = thisPath.inclusiveMatrixInverse();
}

MMatrix MallardViz::localizeSpace(const MMatrix & s) const
{
	MMatrix m = s;
	m *= _worldInverseSpace;
	return m;
}

MMatrix MallardViz::worldizeSpace(const MMatrix & s) const
{
	MMatrix m = s;
	m *= _worldSpace;
	return m;
}

void MallardViz::useActiveView()
{
     _viewport = M3dView::active3dView();
}

char MallardViz::hasDisplayMesh() const
{
	MPlug pm(thisMObject(), ainmesh);
	if(!pm.isConnected())
		return 0;
		
	if(m_bodyMesh == MObject::kNullObj)
		return 0;

	return 1;
}
//:~
