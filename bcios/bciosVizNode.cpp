#include "bciosVizNode.h"
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
#include <TriangleRaster.h>
#include <ConflictGraph.h>
#include <shapeDrawer.h>
#include <fstream> 

MTypeId BCIViz::id( 0x17b733 );
MObject BCIViz::ainput;
MObject BCIViz::outValue;
MObject BCIViz::atargets;

BCIViz::BCIViz() {
m_hitTriangle = 0;
fDriverPos.x = 1.0;
fDriverPos.y = fDriverPos.z = 0.0;
	m_hull = new HullContainer;
	neighbourId[0] = 0;
	neighbourId[1] = 1;
	neighbourId[2] = 2;
}

BCIViz::~BCIViz() {}

MStatus BCIViz::compute( const MPlug& plug, MDataBlock& block )
{
	if( plug == outValue ) {
		MStatus status;
		
		MDagPath path;
		MDagPath::getAPathTo(thisMObject(), path);
		
		MMatrix worldInverseSpace = path.inclusiveMatrixInverse();
		
		MDataHandle inputdata = block.inputValue(ainput, &status);
        if(status) {
			const MMatrix drvSpace = inputdata.asMatrix();
			fDriverPos.x = drvSpace(3, 0);
			fDriverPos.y = drvSpace(3, 1);
			fDriverPos.z = drvSpace(3, 2);
			
			fDriverPos *= worldInverseSpace;
		}
		
		MVector onSphere = fDriverPos;
		onSphere.normalize();
		
		fOnSpherePos = onSphere;
		
		fTargetPositions.clear();
		
		MArrayDataHandle htarget = block.inputArrayValue( atargets );
		unsigned numTarget = htarget.elementCount();
		
		for(unsigned i = 0; i<numTarget; i++) {
			MDataHandle tgtdata = htarget.inputValue(&status);
			if(status) {
				const MMatrix tgtSpace = tgtdata.asMatrix();
				MPoint tgtPos(tgtSpace(3,0), tgtSpace(3,1), tgtSpace(3,2));
				tgtPos *= worldInverseSpace;
				MVector disp = tgtPos;
				disp.normalize();
				tgtPos = disp;
				fTargetPositions.append(tgtPos);
			}
			htarget.next();
		}
		
		if(numTarget < 4)
		{
			MGlobal::displayWarning("convex hull must have no less than 4 targes.");
			return MS::kSuccess;
		}
		
		constructHull();
		
		findNeighbours();
		
		calculateWeight();

        MArrayDataHandle outputHandle = block.outputArrayValue( outValue );
		
		int numWeight = fTargetPositions.length();
		
		MDoubleArray weights;
		weights.setLength(numWeight);
		
		for(int i=0; i < numWeight; i++) 
			weights[i] = 0.0;
			
		weights[neighbourId[0]] = fAlpha;
		weights[neighbourId[1]] = fBeta;
		weights[neighbourId[2]] = fGamma;
		
		MArrayDataBuilder builder(outValue, numWeight, &status);
		
		for(int i=0; i < numWeight; i++) {
			MDataHandle outWeightHandle = builder.addElement(i);
			outWeightHandle.set( weights[i] );
			//MGlobal::displayInfo(MString("wei ") + i + " " + weights[i]);
		}
		
		outputHandle.set(builder);
		outputHandle.setAllClean();
    }

	return MS::kSuccess;
}

char BCIViz::constructHull()
{
	m_hull->destroy();
	const int numTargets = fTargetPositions.length();
	
	for(int i = 0; i < numTargets; i++) 
	{
		Vertex * v = new Vertex;

		v->x = fTargetPositions[i].x;
		v->y = fTargetPositions[i].y;
		v->z = fTargetPositions[i].z;

		m_hull->addVertex(v);
		v->setData((char*)new ConflictGraph(0));
	}
	m_hull->processHull();
	return 1;
}

void BCIViz::draw( M3dView & view, const MDagPath & path, 
							 M3dView::DisplayStyle style,
							 M3dView::DisplayStatus status )
{ 	
	MObject thisNode = thisMObject();
	view.beginGL(); 
	
	glPushAttrib (GL_CURRENT_BIT);
	glDisable(GL_DEPTH_TEST);
	drawDriver();
	drawTargets();
	drawNeighbours();
	glEnable(GL_DEPTH_TEST);
	glPopAttrib();
	view.endGL();
}

bool BCIViz::isBounded() const
{ 
	return true;
}

MBoundingBox BCIViz::boundingBox() const
{   
	
	MPoint corner1(-1.0, -1.0, -1.0);
	MPoint corner2( 1.0,  1.0,  1.0);

	return MBoundingBox( corner1, corner2 );
}

void* BCIViz::creator()
{
	return new BCIViz();
}

MStatus BCIViz::initialize()
{ 
	MFnNumericAttribute numFn;
	MFnMatrixAttribute matAttr;
	MStatus			 stat;
	
	ainput = matAttr.create( "input", "in", MFnMatrixAttribute::kDouble );
	matAttr.setStorable(false);
	matAttr.setWritable(true);
	matAttr.setConnectable(true);
	addAttribute(ainput);
	
	atargets = matAttr.create( "target", "tgt", MFnMatrixAttribute::kDouble );
	matAttr.setStorable(false);
	matAttr.setWritable(true);
	matAttr.setArray(true);
	matAttr.setConnectable(true);
	addAttribute(atargets);
	
	outValue = numFn.create( "outValue", "ov", MFnNumericData::kDouble );
	numFn.setStorable(false);
	numFn.setWritable(false);
	numFn.setReadable(true);
	numFn.setArray(true);
	numFn.setUsesArrayDataBuilder( true );
	addAttribute(outValue);
	
	attributeAffects(ainput, outValue);
	attributeAffects(atargets, outValue);
	return MS::kSuccess;
}

void BCIViz::drawSphere() const
{
	const float angleDelta = 3.14159269f / 36.f;
	float a0, a1, b0, b1;
	glBegin(GL_LINES);
	for(int i=0; i<72; i++) {
		float angleMin = angleDelta * i;
		float angleMax = angleMin + angleDelta;
		
		a0 = cos(angleMin);
		b0 = sin(angleMin);
		
		a1 = cos(angleMax);
		b1 = sin(angleMax);
		
		glVertex3f(a0, b0, 0.f);
		glVertex3f(a1, b1, 0.f);
		
		glVertex3f(a0, 0.f, b0);
		glVertex3f(a1, 0.f, b1);
		
		glVertex3f(0.f, a0, b0);
		glVertex3f(0.f, a1, b1);
	}
	glEnd();
}

void BCIViz::drawDriver() const
{
	const MPoint proof = fTargetPositions[neighbourId[0]] * fAlpha + fTargetPositions[neighbourId[1]] * fBeta + fTargetPositions[neighbourId[2]] * fGamma;
	glBegin(GL_LINES);
	glVertex3f(proof.x, proof.y, proof.z);
	glVertex3f(fDriverPos.x, fDriverPos.y, fDriverPos.z);
	glEnd();
}

void BCIViz::drawTargets() const
{
	if(m_hull->getNumFace() < 4) {
		drawSphere();
		return;
	}
	
	ShapeDrawer drawer;
	drawer.drawWiredFace(m_hull);
	
}

void BCIViz::drawNeighbours() const
{
	Facet f = m_hull->getFacet(m_hitTriangle);
	const Vertex p0 = f.getVertex(0);
	const Vertex p1 = f.getVertex(1);
	const Vertex p2 = f.getVertex(2);
	
	drawCircleAround(p0);
	drawCircleAround(p1);
	drawCircleAround(p2);
	drawCircleAround(m_hitP);
}

void BCIViz::drawCircleAround(const Vector3F& center) const
{
	Vector3F nor(center.x, center.y, center.z);
	Vector3F tangent = nor.perpendicular();
	
	Vector3F v0 = tangent * 0.1f;
	Vector3F p,;
	const float delta = 3.14159269f / 9.f;
	
	glBegin(GL_LINES);
	for(int i = 0; i < 18; i++) {
		p = nor + v0;
		glVertex3f(p.x, p.y, p.z);
		
		v0.rotateAroundAxis(nor, delta);
		
		p = nor + v0;
		glVertex3f(p.x, p.y, p.z);
	}
	glEnd();
}

void BCIViz::findNeighbours()
{
	const int numTri = m_hull->getNumFace();
	if(numTri < 4) {
		MGlobal::displayInfo("barycentric coordinate cannot find three neighbours.");
		return;
	}
	
	Vector3F d;
	d.x = fDriverPos.x;
	d.y = fDriverPos.y;
	d.z = fDriverPos.z;
	
	d.normalize();
	
	m_hitTriangle = 0;
	for(int i = 0; i < numTri; i++)
	{
		Facet f = m_hull->getFacet(i);
		Vertex p0 = f.getVertex(0);
		
		const Vector3F nor = f.getNormal();
		
		float t = p0.dot(nor) / d.dot(nor); 
		if(t < 0.f) continue;
		
		Vertex p1 = f.getVertex(1);
		Vertex p2 = f.getVertex(2);
		
		m_hitTriangle = i;
		m_hitP = d * t; 

		Vector3F e01 = p1 - p0;
		Vector3F e02 = p2 - p0;
		Vector3F tmp = e01.cross(e02);
		if(tmp.dot(nor) < 0.f) {
			Vertex sw = p1;
			p1 = p2;
			p2 = sw;
		}
		
		e01 = p1 - p0;
		Vector3F x0 = m_hitP - p0;
		
		Vector3F e12 = p2 - p1;
		Vector3F x1 = m_hitP - p1;
		
		Vector3F e20 = p0 - p2;
		Vector3F x2 = m_hitP - p2;
		
		neighbourId[0] = p0.getIndex();
		neighbourId[1] = p1.getIndex();
		neighbourId[2] = p2.getIndex();
		
		if(e01.cross(x0).dot(nor) < 0.f) continue;
		if(e12.cross(x1).dot(nor) < 0.f) continue;
		if(e20.cross(x2).dot(nor) < 0.f) continue;
		
		return;
	}
}

void BCIViz::calculateWeight()
{
	TriangleRaster tri;
	Facet f = m_hull->getFacet(m_hitTriangle);
	Vertex p0; 
	p0.x = fTargetPositions[neighbourId[0]].x;
	p0.y = fTargetPositions[neighbourId[0]].y;
	p0.z = fTargetPositions[neighbourId[0]].z;
	Vertex p1;
	p1.x = fTargetPositions[neighbourId[1]].x;
	p1.y = fTargetPositions[neighbourId[1]].y;
	p1.z = fTargetPositions[neighbourId[1]].z;
	Vertex p2;
	p2.x = fTargetPositions[neighbourId[2]].x;
	p2.y = fTargetPositions[neighbourId[2]].y;
	p2.z = fTargetPositions[neighbourId[2]].z;

	tri.create(p0, p1, p2);
	tri.isPointWithin(m_hitP, fAlpha, fBeta, fGamma); 
}
//:~
