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
#include <maya/MFnFreePointTriadManip.h>
#include <maya/MFnNumericData.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnMeshData.h>
#include <CircleCurve.h>
#include <AHelper.h>

MManipData StickyLocatorManip::startPointCallback(unsigned index) const
{
	MFnNumericData numData;
	MObject numDataObj = numData.create(MFnNumericData::k3Double);
	MVector vec = nodeTranslation();
	numData.setData(vec.x, vec.y, vec.z);
	return MManipData(numDataObj);
}


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

    MString manipName("distanceManip");
    MString distanceName("distance");

    MPoint startPoint(0.0, 0.0, 0.0);
    MVector direction(0.0, 1.0, 0.0);
    fDistanceManip = addDistanceManip(manipName,
									  distanceName);
	MFnDistanceManip distanceManipFn(fDistanceManip);
	distanceManipFn.setStartPoint(startPoint);
	distanceManipFn.setDirection(direction);
	
	fDirectionManip = addFreePointTriadManip("freePointTriadManip",
										"point");
	MFnFreePointTriadManip directionManipFn(fDirectionManip);
	
    return stat;
}


MStatus StickyLocatorManip::connectToDependNode(const MObject &node)
{
    MStatus stat;

	// Get the DAG path
	//
	MFnDagNode dagNodeFn(node);
	dagNodeFn.getPath(fNodePath);

    // Connect the plugs
    //    

    MFnDistanceManip distanceManipFn(fDistanceManip);
    MFnDependencyNode nodeFn(node);    

	MPlug sizePlug = nodeFn.findPlug("size", &stat);
    if (MStatus::kFailure != stat) {
	    distanceManipFn.connectToDistancePlug(sizePlug);
		unsigned startPointIndex = distanceManipFn.startPointIndex();
	    addPlugToManipConversionCallback(startPointIndex, 
										 (plugToManipConversionCallback) 
										 &StickyLocatorManip::startPointCallback);
		finishAddingManips();
	    MPxManipContainer::connectToDependNode(node);
	}
	
	MFnFreePointTriadManip directionManipFn;
    directionManipFn.setObject(fDirectionManip);
	MPlug directionPlug = nodeFn.findPlug("displaceVec", &stat);
    if (MStatus::kFailure != stat) {
	    directionManipFn.connectToPointPlug(directionPlug);
	}

    return stat;
}


void StickyLocatorManip::draw(M3dView & view, 
								 const MDagPath &path, 
								 M3dView::DisplayStyle style,
								 M3dView::DisplayStatus status)
{ 
    MPxManipContainer::draw(view, path, style, status);
    view.beginGL(); 

    MPoint textPos = nodeTranslation();
    char str[100];
    sprintf(str, "Stretch Me!"); 
    MString distanceText(str);
    view.drawText(distanceText, textPos, M3dView::kLeft);
	
    view.endGL();
}


MTypeId StickyLocator::id( 0x38d650db );
MObject StickyLocator::size;
MObject StickyLocator::aMoveVX;
MObject StickyLocator::aMoveVY;
MObject StickyLocator::aMoveVZ;
MObject StickyLocator::aMoveV;
MObject StickyLocator::ainmesh;
MObject StickyLocator::avertexId;
MObject StickyLocator::aoutMeanX;
MObject StickyLocator::aoutMeanY;
MObject StickyLocator::aoutMeanZ;
MObject StickyLocator::aoutMean;

StickyLocator::StickyLocator() 
{
	m_circle = new CircleCurve;
	m_origin = MPoint(0.0, 0.0, 0.0);
}


StickyLocator::~StickyLocator() 
{
	delete m_circle;
}


MStatus StickyLocator::compute(const MPlug &plug, MDataBlock &data)
{
	MStatus stat;
	if( plug == aoutMeanX ) {
		MDataHandle hmesh = data.inputValue(ainmesh);
		MObject mesh = hmesh.asMesh();
		if(mesh.isNull()) {
			AHelper::Info<MString>("StickyLocator error no input mesh", plug.name() );
			return MS::kFailure;
		}
		
		MDataHandle hvid = data.inputValue(avertexId);
		int vid = hvid.asInt();
		
		MFnMesh fmesh(mesh, &stat);
		if(!stat) {
			AHelper::Info<MString>("StickyLocator error no mesh fn", plug.name() );
			return stat;
		}
		
		if(vid < 0 || vid >= fmesh.numVertices() ) {
			AHelper::Info<int>("StickyLocator error invalid vertex id", vid );
			return stat;
		}
		
		fmesh.getPoint(vid, m_origin);
		
		MDataHandle outputHandle = data.outputValue( aoutMeanX );
		outputHandle.set( m_origin.x );
		
		data.setClean(plug);
	}
	else if( plug == aoutMeanY ) {
		MDataHandle outputHandle = data.outputValue( aoutMeanY );
		outputHandle.set( m_origin.y );
		data.setClean(plug);
	}
	else if( plug == aoutMeanZ ) {
		MDataHandle outputHandle = data.outputValue( aoutMeanZ );
		outputHandle.set( m_origin.z );
		data.setClean(plug);
	}
	else return MS::kUnknownParameter;
	return MS::kSuccess;
}

void StickyLocator::draw(M3dView &view, const MDagPath &path, 
							M3dView::DisplayStyle style,
							M3dView::DisplayStatus status)
{ 
	// Get the size
	//
	MObject thisNode = thisMObject();
	MPlug meshPlug(thisNode, ainmesh);
	MObject omesh;
	meshPlug.getValue(omesh);
	
	MPlug sizePlug(thisNode, size);
	double sizeVal = sizePlug.asDouble();
	
	MPlug vxPlug(thisNode, aMoveVX);
	double vx = vxPlug.asDouble();
	MPlug vyPlug(thisNode, aMoveVY);
	double vy = vyPlug.asDouble();
	MPlug vzPlug(thisNode, aMoveVZ);
	double vz = vzPlug.asDouble();
	
	view.beginGL(); 
 
		// Push the color settings
		// 
		glPushAttrib(GL_CURRENT_BIT);

		if (status == M3dView::kActive) {
			view.setDrawColor(13, M3dView::kActiveColors);
		} else {
			view.setDrawColor(13, M3dView::kDormantColors);
		}  
	
		glPopAttrib();
		
	glDisable(GL_DEPTH_TEST);
	
	glPushMatrix();
	const float m1[16] = {sizeVal,0,0,0,
					0,sizeVal,0,0,
					0,0,sizeVal,0,
					m_origin.x, m_origin.y, m_origin.z, 1};
	glMultMatrixf(m1);
	drawCircle();
	glPopMatrix();
	
	glBegin(GL_LINES);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(vx, vy, vz);
	glEnd();
	
	glEnable(GL_DEPTH_TEST);
	
	view.endGL();
}

void StickyLocator::drawCircle() const
{
	Vector3F p;
	glBegin(GL_LINE_STRIP);
	for(unsigned i = 0; i < m_circle->numVertices(); i++) {
		p = m_circle->getCv(i);
		glVertex3f(p.x, p.y, p.z);
	}
	glEnd();
}

bool StickyLocator::isBounded() const
{ 
	return true;
}


MBoundingBox StickyLocator::boundingBox() const
{   
	// Get the size
	//
	MObject thisNode = thisMObject();
	MPlug plug(thisNode, size);

	double multiplier = plug.asDouble();
 
	MPoint corner1(-1.0, -1.0, -1.0);
	MPoint corner2(1.0, 1.0, 1.0);

	corner1 = corner1 * multiplier;
	corner2 = corner2 * multiplier;

	return MBoundingBox(corner1, corner2);
}


void* StickyLocator::creator()
{
	return new StickyLocator();
}

MStatus StickyLocator::initialize()
{ 
	MFnNumericAttribute numericFn;
	MFnTypedAttribute typedAttr;
	MStatus			 stat;
	
	aMoveVX = numericFn.create("displaceX", "dspx", 
										 MFnNumericData::kDouble, 0.0, &stat);
	aMoveVY = numericFn.create("displaceY", "dspy",
										 MFnNumericData::kDouble, 0.0, &stat);
	aMoveVZ = numericFn.create("displaceZ", "dspz",
										 MFnNumericData::kDouble, 1.0, &stat);
	aMoveV = numericFn.create("displaceVec", "dspv",
										aMoveVX,
										aMoveVY,
										aMoveVZ, &stat);
	
	stat = addAttribute(aMoveV);
	if (!stat) {
		stat.perror("addAttribute");
		return stat;
	}
	
	size = numericFn.create("size", "sz", MFnNumericData::kDouble);
	numericFn.setDefault(1.0);
	numericFn.setStorable(true);
	numericFn.setWritable(true);
	
	stat = addAttribute(size);
	if (!stat) {
		stat.perror("addAttribute");
		return stat;
	}
	
	ainmesh = typedAttr.create("inMesh", "inm", MFnMeshData::kMesh);
	typedAttr.setStorable(false);
	typedAttr.setWritable(true);
	typedAttr.setConnectable(true);
	addAttribute( ainmesh );
	
	avertexId = numericFn.create("vertexId", "vid", MFnNumericData::kInt);
	numericFn.setDefault(0);
	numericFn.setStorable(true);
	numericFn.setWritable(true);
	addAttribute( avertexId );
	
	aoutMeanX = numericFn.create("meanX", "mnx", 
										 MFnNumericData::kDouble, 0.0, &stat);
	aoutMeanY = numericFn.create("meanY", "mny",
										 MFnNumericData::kDouble, 0.0, &stat);
	aoutMeanZ = numericFn.create("meanZ", "mnz",
										 MFnNumericData::kDouble, 0.0, &stat);
	aoutMean = numericFn.create("outMean", "omn",
										aoutMeanX,
										aoutMeanY,
										aoutMeanZ, &stat);
										
	numericFn.setStorable(false);
	numericFn.setWritable(false);
	addAttribute(aoutMean);
	
	attributeAffects(ainmesh, aoutMean);
	
	MPxManipContainer::addToManipConnectTable(id);

	return MS::kSuccess;
}
