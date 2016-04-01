/*
 *  StickyLocator.cpp
 *  manuka
 *
 *  Created by jian zhang on 1/19/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "StickyLocator.h"
#include <maya/MFnMeshData.h>
#include <maya/MFnIntArrayData.h>
#include <maya/MPxManipContainer.h>
#include <AHelper.h>
#include <linearMath.h>

MTypeId StickyLocator::id( 0x38d650db );
MObject StickyLocator::size;
MObject StickyLocator::aMoveVX;
MObject StickyLocator::aMoveVY;
MObject StickyLocator::aMoveVZ;
MObject StickyLocator::aMoveV;
MObject StickyLocator::ainmesh;
MObject StickyLocator::avertexId;
MObject StickyLocator::ainrefi;
MObject StickyLocator::ainrefd;
MObject StickyLocator::avertexSpace;
MObject StickyLocator::adropoff;

StickyLocator::StickyLocator() 
{
	m_circle = new CircleCurve;
	m_refScale = 0.f;
	m_origin = MPoint(0.0, 0.0, 0.0);
	m_P = new lfr::DenseMatrix<float>();
	m_Q = new lfr::DenseMatrix<float>();
	m_S = new lfr::DenseMatrix<float>(3, 3);
	m_Vd = new lfr::DenseMatrix<float>(3, 3);
	m_Ri = new lfr::DenseMatrix<float>(3, 3);
	m_Ri->setZero();
	m_Ri->addDiagonal(1.f);
	m_scad = new lfr::DenseMatrix<float>(3, 3);
	m_scad->setZero();
	m_svdSolver = new lfr::SvdSolver<float>();
}

StickyLocator::~StickyLocator() 
{
	delete m_circle;
	delete m_P;
	delete m_Q;
	delete m_S;
	delete m_Vd;
	delete m_Ri;
	delete m_scad;
	delete m_svdSolver;
}

MStatus StickyLocator::compute(const MPlug &plug, MDataBlock &data)
{
	MStatus stat;
	if( plug == avertexSpace) {
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
		
		MDataHandle hindices = data.inputValue(ainrefi, &stat);
		MFnIntArrayData findices(hindices.data(), &stat);
		MIntArray indices = findices.array();
		
		if(indices.length() < 1) return MS::kUnknownParameter;
		
		MDataHandle hdiffs = data.inputValue(ainrefd, &stat);
		MFnVectorArrayData fdiffs(hdiffs.data(), &stat);
		MVectorArray diffs = fdiffs.array();
		if(diffs.length() < 1) return MS::kUnknownParameter;
		
		if(indices.length() != diffs.length() ) return MS::kUnknownParameter;
		
		if(m_refScale<1e-6f) buildRefScale(diffs);
		
		updateRotation(fmesh, indices, diffs);
		
		MMatrix m;
		m.matrix[0][0] = m_Ri->column(0)[0];
		m.matrix[0][1] = m_Ri->column(0)[1];
		m.matrix[0][2] = m_Ri->column(0)[2];
		m.matrix[1][0] = m_Ri->column(1)[0];
		m.matrix[1][1] = m_Ri->column(1)[1];
		m.matrix[1][2] = m_Ri->column(1)[2];
		m.matrix[2][0] = m_Ri->column(2)[0];
		m.matrix[2][1] = m_Ri->column(2)[1];
		m.matrix[2][2] = m_Ri->column(2)[2];
		m.matrix[3][0] = m_origin.x;
		m.matrix[3][1] = m_origin.y;
		m.matrix[3][2] = m_origin.z;
		
		MDataHandle outputHandle = data.outputValue( avertexSpace );
		
		outputHandle.set( m );
		
		data.setClean(plug);
	}
	else return MS::kUnknownParameter;
	return MS::kSuccess;
}

void StickyLocator::draw(M3dView &view, const MDagPath &path, 
							M3dView::DisplayStyle style,
							M3dView::DisplayStatus status)
{ 
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
 
	glPushAttrib(GL_CURRENT_BIT);

	if (status == M3dView::kActive) {
		view.setDrawColor(13, M3dView::kActiveColors);
	} else {
		view.setDrawColor(13, M3dView::kDormantColors);
	}  

	glPopAttrib();
		
	glDisable(GL_DEPTH_TEST);
	
	glPushMatrix();
	const float m0[16] = {m_Ri->column(0)[0],m_Ri->column(0)[1],m_Ri->column(0)[2],0,
					m_Ri->column(1)[0],m_Ri->column(1)[1],m_Ri->column(1)[2],0,
					m_Ri->column(2)[0],m_Ri->column(2)[1],m_Ri->column(2)[2],0,
					m_origin.x, m_origin.y, m_origin.z, 1};
	glMultMatrixf(m0);
	
	glPushMatrix();
	const float m1[16] = {sizeVal,0,0,0,
					0,sizeVal,0,0,
					0,0,sizeVal,0,
					0,0,0, 1};
	glMultMatrixf(m1);
	drawCircle();
	glPopMatrix();
	
	glPushMatrix();
	const float m2[16] = {0,0,-sizeVal,0,
					0,sizeVal,0,0,
					sizeVal,0,0,0,
					0,0,0, 1};
	glMultMatrixf(m2);
	drawCircle();
	glPopMatrix();
	
	glPushMatrix();
	const float m3[16] = {sizeVal,0,0,0,
					0,0,-sizeVal,0,
					0,sizeVal,0,0,
					0,0,0, 1};
	glMultMatrixf(m3);
	drawCircle();
	glPopMatrix();
	
	glBegin(GL_LINES);
	glVertex3f(0,0,0);
	glVertex3f(vx, vy, vz);
	glEnd();
	
	glPopMatrix();
	
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
	float sca = m_scad->column(0)[0];
 
	MPoint corner1(-1.0, -1.0, -1.0);
	MPoint corner2(1.0, 1.0, 1.0);

	corner1 = m_origin + corner1 * multiplier * sca;
	corner2 = m_origin + corner2 * multiplier * sca;

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
										 MFnNumericData::kDouble);
	aMoveVY = numericFn.create("displaceY", "dspy",
										 MFnNumericData::kDouble);
	aMoveVZ = numericFn.create("displaceZ", "dspz",
										 MFnNumericData::kDouble);
	aMoveV = numericFn.create("displaceVec", "dspv",
										aMoveVX,
										aMoveVY,
										aMoveVZ, &stat);
	numericFn.setDefault(0.0, 0.0, 1.0);
	numericFn.setKeyable(true);
	
	stat = addAttribute(aMoveV);
	if (!stat) {
		stat.perror("addAttribute");
		return stat;
	}
	
	size = numericFn.create("size", "sz", MFnNumericData::kDouble);
	numericFn.setDefault(1.0);
	numericFn.setStorable(true);
	numericFn.setWritable(true);
	numericFn.setKeyable(true);
	
	stat = addAttribute(size);
	if (!stat) {
		stat.perror("addAttribute");
		return stat;
	}
	
	adropoff = numericFn.create("dropoff", "dpo", MFnNumericData::kDouble);
	numericFn.setDefault(1.0);
	numericFn.setStorable(true);
	numericFn.setWritable(true);
	numericFn.setKeyable(true);
	
	stat = addAttribute(adropoff);
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
	
	MIntArray defaultIntArray;
	MFnIntArrayData intArrayDataFn;
	intArrayDataFn.create( defaultIntArray );
	
	ainrefi = typedAttr.create("refInds", "rids",
											MFnData::kIntArray,
											intArrayDataFn.object(),
											&stat );
												
	if(!stat) MGlobal::displayWarning("failed create ref id attr");
	typedAttr.setStorable(true);
	addAttribute(ainrefi);
	
	MVectorArray defaultVectArray;
	MFnVectorArrayData vectArrayDataFn;
	vectArrayDataFn.create( defaultVectArray );
	
	ainrefd = typedAttr.create("refDisplace", "rdp", MFnData::kVectorArray,
											vectArrayDataFn.object(),
											&stat );
	if(!stat) MGlobal::displayWarning("failed create ref dp attr");
	typedAttr.setStorable(true);
	addAttribute(ainrefd);
	
	MFnMatrixAttribute matAttr;
	avertexSpace = matAttr.create( "vertexMatrix", "vtm", MFnMatrixAttribute::kDouble );
 	matAttr.setStorable(false);
	matAttr.setWritable(true);
	matAttr.setConnectable(true);
	addAttribute(avertexSpace);
	attributeAffects(ainmesh, avertexSpace);
	
	MPxManipContainer::addToManipConnectTable(id);

	return MS::kSuccess;
}

void StickyLocator::buildRefScale(const MVectorArray & diffs)
{
	const int n = diffs.length();
	m_P->resize(3, n);
	m_Q->resize(3, n);
	
	MPoint pt;
	for(int i=0; i<n; ++i ) {
		// fmesh.getPoint(indices[i], pt);
		MVector dp = diffs[i];
		m_P->column(i)[0] = dp.x;
		m_P->column(i)[1] = dp.y;
		m_P->column(i)[2] = dp.z;
		
		m_Q->column(i)[0] = dp.x;
		m_Q->column(i)[1] = dp.y;
		m_Q->column(i)[2] = dp.z;
	}
	
	m_P->multTrans(*m_S, *m_Q);
	
	m_svdSolver->compute(*m_S);
	
	m_refScale = m_svdSolver->S().v()[0];
	
	AHelper::Info<float>("refscale", m_refScale);
}

void StickyLocator::updateRotation(const MFnMesh & fmesh, 
					const MIntArray & indices, 
					const MVectorArray & diffs)
{
	const int n = diffs.length();
	m_P->resize(3, n);
	m_Q->resize(3, n);
	
	MPoint pt;
	for(int i=0; i<n; ++i ) {
		MVector dpr = diffs[i];
		m_P->column(i)[0] = dpr.x;
		m_P->column(i)[1] = dpr.y;
		m_P->column(i)[2] = dpr.z;
		
		fmesh.getPoint(indices[i], pt);
		MVector dpc = pt - m_origin;
		m_Q->column(i)[0] = dpc.x;
		m_Q->column(i)[1] = dpc.y;
		m_Q->column(i)[2] = dpc.z;
	}
	
	m_P->multTrans(*m_S, *m_Q);
	
	m_svdSolver->compute(*m_S);
	
	const float d = m_svdSolver->S().v()[0] / m_refScale;
	m_scad->column(0)[0] = d;
	m_scad->column(1)[1] = d;
	m_scad->column(2)[2] = d;
	
	m_svdSolver->Vt().transMult(*m_Vd, * m_scad);
	m_Vd->multTrans(*m_Ri, m_svdSolver->U());
}
