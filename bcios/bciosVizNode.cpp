/*
 createNode barycentricInterpolationViz;
 spaceLocator;
 spaceLocator;
 spaceLocator;
 spaceLocator;
 spaceLocator;
 spaceLocator;
 spaceLocator;
 connectAttr -f locator1.worldMatrix[0] barycentricInterpolationViz1.input;
 connectAttr -f barycentricInterpolationViz1.ov locator2.tx;
 connectAttr -f locator3.worldMatrix[0] barycentricInterpolationViz1.tgt[0];
 connectAttr -f locator4.worldMatrix[0] barycentricInterpolationViz1.tgt[1];
 connectAttr -f locator5.worldMatrix[0] barycentricInterpolationViz1.tgt[2];
 connectAttr -f locator6.worldMatrix[0] barycentricInterpolationViz1.tgt[3];
 connectAttr -f locator7.worldMatrix[0] barycentricInterpolationViz1.tgt[4];
*/

#include <Vector2F.h>
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
#include <fstream> 

MTypeId BCIViz::id( 0x17b733 );
MObject BCIViz::ainput;
MObject BCIViz::outValue;
MObject BCIViz::atargets;

BCIViz::BCIViz() {

fDriverPos.x = 1.0;
fDriverPos.y = fDriverPos.z = 0.0;
			
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
		
		findNeighbours();

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

void BCIViz::draw( M3dView & view, const MDagPath & path, 
							 M3dView::DisplayStyle style,
							 M3dView::DisplayStatus status )
{ 	
	MObject thisNode = thisMObject();
	view.beginGL(); 
	
	glPushAttrib (GL_CURRENT_BIT);
	glDisable(GL_DEPTH_TEST);
	drawSphere();
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
	glBegin(GL_LINES);
	glVertex3f(fHitPos.x, fHitPos.y, fHitPos.z);
	glVertex3f(fDriverPos.x, fDriverPos.y, fDriverPos.z);
	glEnd();
}

void BCIViz::drawTargets() const
{
	const int numTargets = fTargetPositions.length();
	if(numTargets < 1)
		return;
	glBegin(GL_LINES);	
	for(int i = 0; i < numTargets; i++) {
		glVertex3f(0.f, 0.f, 0.f);
		glVertex3f(fTargetPositions[i].x, fTargetPositions[i].y, fTargetPositions[i].z);
	}
	glEnd();
}

void BCIViz::drawNeighbours() const
{
	drawCircleAround(fTargetPositions[neighbourId[0]]);
	drawCircleAround(fTargetPositions[neighbourId[1]]);
	drawCircleAround(fTargetPositions[neighbourId[2]]);
	drawCircleAround(fHitPos);
	
	const MPoint proof = fTargetPositions[neighbourId[0]] * fAlpha + fTargetPositions[neighbourId[1]] * fBeta + fTargetPositions[neighbourId[2]] * fGamma;
	glBegin(GL_LINES);
	glVertex3f(proof.x, proof.y, proof.z);
	glVertex3f(fNeighbours[0].x, fNeighbours[0].y, fNeighbours[0].z);
	glVertex3f(proof.x, proof.y, proof.z);
	glVertex3f(fNeighbours[1].x, fNeighbours[1].y, fNeighbours[1].z);
	glVertex3f(proof.x, proof.y, proof.z);
	glVertex3f(fNeighbours[2].x, fNeighbours[2].y, fNeighbours[2].z);
	glEnd();
}

void BCIViz::drawCircleAround(const MPoint& center) const
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
	const int numTargets = fTargetPositions.length();
	if(numTargets < 3)
		return;
	
	MVector disp;
	float min = 1000.f;
	int a = 0;
	for(int i = 0; i < numTargets; i++) {
		disp = fTargetPositions[i] - fOnSpherePos;
		if(min > disp.length()) {
			fNeighbours[0] = fTargetPositions[i];
			min = disp.length();
			a = i;
		}		
	}
	
	int b = 0;
	if(!secondNeightour(a, b)) {
		MGlobal::displayWarning("BCI cannot find 2nd hit");
		neighbourId[0] = neighbourId[1] = neighbourId[2] = a;
		fAlpha = 1.f;
		fBeta = fGamma = 0.f;
		return;
	}

	int c = 0;
	if(thirdNeighbour(a, b, c)) {
		neighbourId[0] = a;
		neighbourId[1] = b;
		neighbourId[2] = c;
		//MGlobal::displayInfo(MString("nearest ") + a + " " + b + " " + c);
	}
	else {
		MGlobal::displayWarning("BCI cannot find hit within any triangles");
		neighbourId[0] = neighbourId[1] = neighbourId[2] = a;
		fAlpha = 1.f;
		fBeta = fGamma = 0.f;
	}
		
	fNeighbours[2] = fTargetPositions[c];
}

char BCIViz::secondNeightour(int a, int & b)
{
	char res = 0;
	MVector edge0 = fNeighbours[0] - fOnSpherePos;
	float min = 1000.f;
	const int numTargets = fTargetPositions.length();
	for(int i = 0; i < numTargets; i++) {
		if(i==a)
			continue;
			
		MVector edge1 = fTargetPositions[i] - fOnSpherePos;
		float dist = edge1.length(); 
		if(min > dist && edge0 * edge1 < 0.0) {
			fNeighbours[1] = fTargetPositions[i];
			min = dist;
			b = i;
			res = 1;
		}		
	}
	return res;
}

char BCIViz::thirdNeighbour(int a, int b, int &c)
{
	const MVector edge0 = fNeighbours[1] - fNeighbours[0];
	char res = 0;
	const int numTargets = fTargetPositions.length();
	TriangleRaster tri;
	float alpha, beta, gamma;
	
	Vector3F p0(fNeighbours[1].x, fNeighbours[1].y, fNeighbours[1].z);
	Vector3F p1(fNeighbours[0].x, fNeighbours[0].y, fNeighbours[0].z);
	
	double max = 0.0;
	for(int i = 0; i < numTargets; i++) {
		if(i != a && i != b) {
			const MVector edge1 = fTargetPositions[i] - fNeighbours[0];
			MVector N = edge0 ^ edge1;
			N.normalize();
			
			if(N * MVector(fOnSpherePos) < 0.0)
				N *= -1.0;
				
			const double d = MVector(fNeighbours[0]) * N * (-1.0);
			double facing = MVector(fOnSpherePos) * N;
			if(facing < 10e-6) 
				continue;
				
			double t = -d / facing;
			if(t < 0.0 || t > 1.01) 
				continue;
				
			Vector3F q(fOnSpherePos.x * t, fOnSpherePos.y * t, fOnSpherePos.z * t);
			Vector3F p2(fTargetPositions[i].x, fTargetPositions[i].y, fTargetPositions[i].z);
			tri.create(p0, p1, p2);
			
			if(tri.isPointWithin(q, alpha, beta, gamma)) {
				if(t > max) {
					res = 1;
					fHitPos = fOnSpherePos * t;
					max = t;
					c = i;
					fAlpha = beta;
					fBeta = alpha;
					fGamma = gamma;
				}
			}
		}
	}

	return res;
}
//:~
