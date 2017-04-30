#include "StickyManip.h"
#include <maya/MString.h> 
#include <maya/MVector.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MColor.h>
#include <maya/M3dView.h>
#include <maya/MDistance.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MFnDistanceManip.h> 
#include <maya/MFnDirectionManip.h>
#include <maya/MFnNumericData.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnMeshData.h>
#include <maya/MFnMatrixData.h>
#include <maya/MFnIntArrayData.h>
#include <AHelper.h>

using namespace aphid;

MVector StickyLocatorManip::nodeTranslation() const
{
	MFnDagNode dagFn(fNodePath);
	MDagPath path;
	dagFn.getPath(path);
	path.pop();  // pop from the shape to the transform
	MFnTransform transformFn(path);
	return transformFn.translation(MSpace::kWorld);
}

MTypeId StickyLocatorManip::id( 0x38575951 );

StickyLocatorManip::StickyLocatorManip() 
{ 
    // Do not call createChildren from here 
}


StickyLocatorManip::~StickyLocatorManip() 
{
}


void* StickyLocatorManip::creator()
{
     return new StickyLocatorManip();
}


MStatus StickyLocatorManip::initialize()
{ 
    MStatus stat;
    stat = MPxManipContainer::initialize();
    return stat;
}


MStatus StickyLocatorManip::createChildren()
{
    MStatus stat = MStatus::kSuccess;

    fDistanceManip = addDistanceManip("distanceManip",
									  "distance");
	MFnDistanceManip distanceManipFn(fDistanceManip);
	distanceManipFn.setDirection(MVector(0.0, 1.0, 0.0));
	
	fDirectionManip = addDirectionManip("dispaceVecManip",
										"direction");
	
	fDropoffManip = addDistanceManip("dropoffManip",
									  "distance1");
	MFnDistanceManip dropoffManipFn(fDropoffManip);
	dropoffManipFn.setDirection(MVector(1.0, 0.0, 0.0));
	return stat;
}

MStatus StickyLocatorManip::connectToDependNode(const MObject &node)
{
    MStatus stat;

	MFnDagNode dagNodeFn(node);
	dagNodeFn.getPath(fNodePath);

	AHelper::Info<MString>("sticky manip connect to", fNodePath.fullPathName() );
	
	MPlug spacePlug = dagNodeFn.findPlug("vertexMatrix");
	MObject om;
	spacePlug.getValue(om);
	MFnMatrixData fm(om);
	m_rotateM = fm.matrix();
	m_startPOffset = MVector(m_rotateM[3][0], m_rotateM[3][1], m_rotateM[3][2]);
	m_scalingF = MVector(m_rotateM[0][0], m_rotateM[0][1], m_rotateM[0][2]).length();
	
	MFnDependencyNode nodeFn(node);
	
    MFnDistanceManip sizeManipFn(fDistanceManip);
	sizeManipFn.setScalingFactor(m_scalingF);
	
	MPlug sizePlug = nodeFn.findPlug("size", &stat);
    if (MStatus::kFailure != stat) {
	    sizeManipFn.connectToDistancePlug(sizePlug);
		unsigned startPointIndex = sizeManipFn.startPointIndex();
	    addPlugToManipConversionCallback(startPointIndex, 
										 (plugToManipConversionCallback) 
										 &StickyLocatorManip::startPointCallback);
		
	}
	
	double currentSize = 1.0;
	sizePlug.getValue(currentSize);

	MFnDistanceManip dropoffManipFn(fDropoffManip);
	dropoffManipFn.setScalingFactor(m_scalingF * currentSize);
	
	MPlug dropoffPlug = nodeFn.findPlug("dropoff", &stat);
    if (MStatus::kFailure != stat) {
	    dropoffManipFn.connectToDistancePlug(dropoffPlug);
		unsigned startPointIndex = dropoffManipFn.startPointIndex();
	    addPlugToManipConversionCallback(startPointIndex, 
										 (plugToManipConversionCallback) 
										 &StickyLocatorManip::startPointCallback);
	}
	
	MFnDirectionManip directionManipFn(fDirectionManip);
	directionManipFn.setNormalizeDirection(false);
	
	MPlug directionPlug = nodeFn.findPlug("displaceVec", &stat);
	m_localVPlug = directionPlug;
    if (MStatus::kFailure != stat) {
/// initial direction
		MObject ov;
		directionPlug.getValue(ov);
		MFnNumericData flocalV(ov);
		flocalV.getData(m_localV.x, m_localV.y, m_localV.z);
		MVector v = m_localV * m_rotateM;
		directionManipFn.setDirection(v);

/// use callbacks, no connection
		unsigned startPointIndex = directionManipFn.startPointIndex();
	    addPlugToManipConversionCallback(startPointIndex, 
										 (plugToManipConversionCallback) 
										 &StickyLocatorManip::startPointCallback);
		
		unsigned endPointIndex = directionManipFn.endPointIndex();
	    addPlugToManipConversionCallback(endPointIndex, 
										 (plugToManipConversionCallback) 
										 &StickyLocatorManip::endPointCallback);
					
		m_dirPlugIndex = addManipToPlugConversionCallback(directionPlug, 
										 (manipToPlugConversionCallback) 
										 &StickyLocatorManip::endPointCallbackTo);
	}
	
	finishAddingManips();
	return MPxManipContainer::connectToDependNode(node);
}


void StickyLocatorManip::draw(M3dView & view, 
								 const MDagPath &path, 
								 M3dView::DisplayStyle style,
								 M3dView::DisplayStatus status)
{ 
    MPxManipContainer::draw(view, path, style, status);
	return;
    view.beginGL(); 

    MPoint textPos = nodeTranslation();
    char str[100];
    sprintf(str, "Stretch Me!"); 
    MString distanceText(str);
    view.drawText(distanceText, textPos, M3dView::kLeft);
	
    view.endGL();
}

MManipData StickyLocatorManip::startPointCallback(unsigned index) const
{
	MFnNumericData numData;
	MObject numDataObj = numData.create(MFnNumericData::k3Double);
/// use vertex matrix translate
	numData.setData(m_startPOffset.x, m_startPOffset.y, m_startPOffset.z);
	return MManipData(numDataObj);
}

MManipData StickyLocatorManip::endPointCallback(unsigned index)
{
	MFnNumericData numData;
	MObject numDataObj = numData.create(MFnNumericData::k3Double);
	
	MObject ov;
	m_localVPlug.getValue(ov);
	MFnNumericData flocalV(ov);
	flocalV.getData(m_localV.x, m_localV.y, m_localV.z);
/// to world space
	MVector v = m_localV * m_rotateM + m_startPOffset;

	numData.setData(v.x, v.y, v.z);
	return MManipData(numDataObj);
}

MManipData StickyLocatorManip::endPointCallbackTo(unsigned index)
{
	MFnNumericData numData;
	MObject numDataObj = numData.create(MFnNumericData::k3Double);
	
	if(index != m_dirPlugIndex) {
		numData.setData(0.0, 0.0, 0.0);
		return MManipData(numDataObj);
	}
	
	MFnDirectionManip directionManipFn(fDirectionManip);
	
	MVector endP;
	getConverterManipValue (directionManipFn.directionIndex(), endP);
/// to local space	
	MVector v = endP * m_rotateM.inverse();
	m_localV = v;

	numData.setData(v.x, v.y, v.z);
	
	MObject ov = numData.object();
	return MManipData(numDataObj);
}
