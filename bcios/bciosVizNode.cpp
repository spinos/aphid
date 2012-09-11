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
fDriverPos.x = 1.0;
fDriverPos.y = fDriverPos.z = 0.0;
	m_hull = new HullContainer;
}

BCIViz::~BCIViz() {}

char BCIViz::checkHull() const
{
	return (m_hull->getNumFace() > 3 && m_hull->getNumFace() > m_hitTriangle);
}

char BCIViz::checkTarget() const
{
	return (fTargetPositions.length() > 3 && fTargetPositions.length() > neighbourId[0] && fTargetPositions.length() > neighbourId[1] && fTargetPositions.length() > neighbourId[2]);
}

char BCIViz::checkFirstFour(const MPointArray & p) const
{
	const MVector e01 = p[1] - p[0];
	const MVector e02 = p[2] - p[0];
	MVector nor = e01^e02;
	nor.normalize();
	const float d = -(MVector(p[0])*nor); // P.N + d = 0 , P = P0 + Vt, (P0 + Vt).N + d = 0, t = -(P0.N + d)/(V.N)
	const float t = MVector(p[3])*nor + d; // where V = -N
	if(t > -10e-5 && t < 10e-5) return 0;
	return 1;
}

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
		
		fTargetPositions.clear();
		
		MArrayDataHandle htarget = block.inputArrayValue( atargets );
		unsigned numTarget = htarget.elementCount();
		
		fTargetPositions.setLength(numTarget);
		
		for(unsigned i = 0; i<numTarget; i++) {
			MDataHandle tgtdata = htarget.inputValue(&status);
			if(status) {
				const MMatrix tgtSpace = tgtdata.asMatrix();
				MPoint tgtPos(tgtSpace(3,0), tgtSpace(3,1), tgtSpace(3,2));
				tgtPos *= worldInverseSpace;
				MVector disp = tgtPos;
				disp.normalize();
				tgtPos = disp;
				fTargetPositions[i] = tgtPos;
			}
			htarget.next();
		}
		
		m_hitTriangle = 0;
		neighbourId[0] = 0;
		neighbourId[1] = 1;
		neighbourId[2] = 2;
		
		if(!checkTarget())
		{
			MGlobal::displayWarning("convex hull must have no less than 4 targes.");
			return MS::kSuccess;
		}
		
		if(!checkFirstFour(fTargetPositions))
		{
			MGlobal::displayWarning("first 4 targes cannot sit on the same plane.");
			return MS::kSuccess;
		}
		
		if(!constructHull())
		{
			MGlobal::displayWarning("convex hull failed on construction.");
			return MS::kSuccess;
		}

		findNeighbours();
		
		calculateWeight();

        MArrayDataHandle outputHandle = block.outputArrayValue( outValue );
		
		int numWeight = fTargetPositions.length();

		m_resultWeights.setLength(numWeight);
		
		for(int i=0; i < numWeight; i++) 
			m_resultWeights[i] = 0.0;
			
		m_resultWeights[neighbourId[0]] = fAlpha;
		m_resultWeights[neighbourId[1]] = fBeta;
		m_resultWeights[neighbourId[2]] = fGamma;
		
		MArrayDataBuilder builder(outValue, numWeight, &status);
		
		for(int i=0; i < numWeight; i++) {
			MDataHandle outWeightHandle = builder.addElement(i);
			outWeightHandle.set( m_resultWeights[i] );
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
	return m_hull->getNumFace() >= 4;
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
	drawWeights();
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

void BCIViz::drawDriver() const
{
	if(!checkTarget()) return;
	
	const MPoint proof = fTargetPositions[neighbourId[0]] * fAlpha + fTargetPositions[neighbourId[1]] * fBeta + fTargetPositions[neighbourId[2]] * fGamma;
	glBegin(GL_LINES);
	glVertex3f(proof.x, proof.y, proof.z);
	glVertex3f(fDriverPos.x, fDriverPos.y, fDriverPos.z);
	glEnd();
}

void BCIViz::drawTargets() const
{
	ShapeDrawer drawer;
	
	if(!checkHull()) {
		drawer.drawSphere();
		return;
	}
	
	drawer.drawWiredFace(m_hull);
}

void BCIViz::drawNeighbours() const
{
	if(!checkHull()) return;
	
	Facet f = m_hull->getFacet(m_hitTriangle);
	const Vertex p0 = f.getVertex(0);
	const Vertex p1 = f.getVertex(1);
	const Vertex p2 = f.getVertex(2);
	
	ShapeDrawer drawer;
	drawer.drawCircleAround(p0);
	drawer.drawCircleAround(p1);
	drawer.drawCircleAround(p2);
	drawer.drawCircleAround(m_hitP);
}

void BCIViz::drawWeights() const
{
    const int numWeight = fTargetPositions.length();
    
    if(numWeight < 4) return;
	
    glBegin(GL_LINES);
    for(int i=0; i < numWeight; i++) 
    {
        glVertex3f(2.f + i, 0.f, 0.f);
        glVertex3f(2.f + i, m_resultWeights[i] + 0.1f, 0.f);
	}
	glEnd();	
	
}

void BCIViz::findNeighbours()
{
	if(!checkHull()) return;
	const int numTri = m_hull->getNumFace();
	
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
		
		float ddotn = d.dot(nor);
		
		if(ddotn < 10e-5 && ddotn > -10e-5) continue;
		
		float t = p0.dot(nor) / ddotn; 
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
	if(!checkTarget() || !checkHull()) return;
	
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
