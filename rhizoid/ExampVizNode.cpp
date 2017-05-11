/*
 *  ExampVizNode.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 2/5/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "ExampVizNode.h"
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnPointArrayData.h>
#include <maya/MFloatVector.h>
#include <maya/MFnPluginData.h>
#include <maya/MFnVectorArrayData.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnEnumAttribute.h>
#include <ExampData.h>
#include <math/linearMath.h>
#include <geom/ConvexShape.h>
#include <sdb/VectorArray.h>
#include <sdb/ValGrid.h>
#include <AllMama.h>

MTypeId ExampViz::id( 0x95a20e );
MObject ExampViz::abboxminv;
MObject ExampViz::abboxmaxv;
MObject ExampViz::adoplen;
MObject ExampViz::adopPBuf;
MObject ExampViz::adopNBuf;
MObject ExampViz::adopCBuf;
MObject ExampViz::adrawColor;
MObject ExampViz::adrawColorR;
MObject ExampViz::adrawColorG;
MObject ExampViz::adrawColorB;
MObject ExampViz::adrawDopSizeX;
MObject ExampViz::adrawDopSizeY;
MObject ExampViz::adrawDopSizeZ;
MObject ExampViz::adrawDopSize;
MObject ExampViz::aradiusMult;
MObject ExampViz::aininstspace;
MObject ExampViz::avoxactive;
MObject ExampViz::avoxvisible;
MObject ExampViz::avoxpriority;
MObject ExampViz::adrawVoxTag;
MObject ExampViz::outValue;

using namespace aphid;

ExampViz::ExampViz()
{
	m_preDiffCol[0] = 0.f;
	m_preDiffCol[1] = 0.f;
	m_preDiffCol[2] = 0.f;
}

ExampViz::~ExampViz() 
{}

MStatus ExampViz::compute( const MPlug& plug, MDataBlock& block )
{
	if( plug == outValue ) {
		
		MDataHandle radiusMultH = block.inputValue(aradiusMult);
		float radiusScal = radiusMultH.asFloat();
		setGeomSizeMult(radiusScal);
		
		BoundingBox bb;
		
		MDataHandle bbminH = block.inputValue(abboxminv);
		MFloatVector& vmin = bbminH.asFloatVector();
		bb.setMin(vmin.x, vmin.y, vmin.z);
		
		MDataHandle bbmaxH = block.inputValue(abboxmaxv);
		MFloatVector& vmax = bbmaxH.asFloatVector();
		bb.setMax(vmax.x, vmax.y, vmax.z);

		setGeomBox(&bb);
		
		MDataHandle drszx = block.inputValue(adrawDopSizeX);
		MDataHandle drszy = block.inputValue(adrawDopSizeY);
		MDataHandle drszz = block.inputValue(adrawDopSizeZ);
		setDopSize(drszx.asFloat(), drszy.asFloat(), drszz.asFloat() );
	
		float diffCol[3];
		diffCol[0] = block.inputValue(adrawColorR).asFloat(); 
		diffCol[1] = block.inputValue(adrawColorG).asFloat(); 
		diffCol[2] = block.inputValue(adrawColorB).asFloat();
		setDiffuseMaterialCol(diffCol);
		
		const bool vis = block.inputValue(avoxvisible).asBool(); 
		setVisible(vis);
		
		if(!loadTriangles(block) ) {
			AHelper::Info<MString>(" ERROR ExampViz has no draw data", MFnDependencyNode(thisMObject() ).name() );
		}
		
		updateGridUniformColor(diffCol);
		
		MDataHandle detailTypeHandle = block.inputValue( adrawVoxTag );
		const short detailType = detailTypeHandle.asShort();
		setDetailDrawType(detailType);
		
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

void ExampViz::draw( M3dView & view, const MDagPath & path, 
							 M3dView::DisplayStyle style,
							 M3dView::DisplayStatus status )
{
	MObject selfNode = thisMObject();
	updateGeomBox(selfNode);
	
	MPlug rPlug(selfNode, adrawColorR);
	MPlug gPlug(selfNode, adrawColorG);
	MPlug bPlug(selfNode, adrawColorB);
	
	float * diffCol = diffuseMaterialColV();
	diffCol[0] = rPlug.asFloat();
	diffCol[1] = gPlug.asFloat();
	diffCol[2] = bPlug.asFloat();
	
	MPlug szxp(selfNode, adrawDopSizeX);
	MPlug szyp(selfNode, adrawDopSizeY);
	MPlug szzp(selfNode, adrawDopSizeZ);
	setDopSize(szxp.asFloat(), szyp.asFloat(), szzp.asFloat() );
	
	bool stat = pntBufLength() > 0;
	if(!stat) {
		stat = loadTriangles(selfNode);
	}
	if(!stat) {
		AHelper::Info<MString>(" ERROR ExampViz has no draw data", MFnDependencyNode(selfNode).name() );
		return;
	}
	
	updateGridUniformColor(diffCol);
	
	MPlug detailTypePlg(selfNode, adrawVoxTag);
	const short detailType = detailTypePlg.asShort();
	setDetailDrawType(detailType);
	
	MDagPath cameraPath;
	view.getCamera(cameraPath);
	Matrix33F mf;
	AHelper::GetViewMatrix(&mf, cameraPath);
	mf *= geomSize();
    mf.glMatrix(m_transBuf);
	
	view.beginGL();
	
	const BoundingBox & bbox = geomBox();
	drawBoundingBox(&bbox);
	
	drawZCircle(m_transBuf);
	
	glPushAttrib(GL_LIGHTING_BIT | GL_CURRENT_BIT);

	glColor3fv(diffCol);
	
	glPointSize(2.f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glEnableClientState(GL_VERTEX_ARRAY);
	
	drawAWireDop();
	
	glDisable(GL_LIGHTING);
	//if ( style == M3dView::kFlatShaded || 
	//	    style == M3dView::kGouraudShaded ) {
		
		glEnableClientState(GL_COLOR_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);
		
		if(detailType == 0) {
			drawPoints();
		} else {
			drawSolidGrid();
		}
		
		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);
	
	//} else {
	//	drawWiredPoints();
	//} 
	glDisableClientState(GL_VERTEX_ARRAY);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glPopAttrib();
	
	view.endGL();
}

bool ExampViz::isBounded() const
{ return true; }

MBoundingBox ExampViz::boundingBox() const
{   
	const BoundingBox & bbox = geomBox();
	
	MPoint corner1(bbox.m_data[0], bbox.m_data[1], bbox.m_data[2]);
	MPoint corner2(bbox.m_data[3], bbox.m_data[4], bbox.m_data[5]);

	return MBoundingBox( corner1, corner2 );
}

void* ExampViz::creator()
{
	return new ExampViz();
}

MStatus ExampViz::initialize()
{ 
	MFnNumericAttribute numFn;
	MFnTypedAttribute typedFn;
	MFnEnumAttribute	enumAttr;
	
	MStatus			 stat;
	
	adrawColorR = numFn.create( "dspColorR", "dspr", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.47f);
	numFn.setMin(0.f);
	numFn.setMax(1.f);
	addAttribute(adrawColorR);
	
	adrawColorG = numFn.create( "dspColorG", "dspg", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.46f);
	numFn.setMin(0.f);
	numFn.setMax(1.f);
	addAttribute(adrawColorG);
	
	adrawColorB = numFn.create( "dspColorB", "dspb", MFnNumericData::kFloat);
	numFn.setStorable(true);
	numFn.setKeyable(true);
	numFn.setDefault(0.45f);
	numFn.setMin(0.f);
	numFn.setMax(1.f);
	addAttribute(adrawColorB);
	
	adrawColor = numFn.create( "dspColor", "dspc", adrawColorR, adrawColorG, adrawColorB );
	numFn.setStorable(true);
	numFn.setKeyable(true);
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
	
	abboxminv = numFn.create( "BBoxMin", "bbxmn", MFnNumericData::k3Float );
	numFn.setStorable(true);
	numFn.setDefault(-1.f, -1.f, -1.f);
	addAttribute(abboxminv);
	
	abboxmaxv = numFn.create( "BBoxMax", "bbxmx", MFnNumericData::k3Float );
	numFn.setStorable(true);
	numFn.setDefault(1.f, 1.f, 1.f);
	addAttribute(abboxmaxv);
	
	adrawVoxTag = enumAttr.create( "dspDetailType", "ddt", 0, &stat );
	enumAttr.addField( "point", 0 );
	enumAttr.addField( "grid", 1 );
	enumAttr.setHidden( false );
	enumAttr.setKeyable( true );
	addAttribute(adrawVoxTag);
	
	outValue = typedFn.create( "outValue", "ov", MFnData::kPlugin );
	typedFn.setStorable(false);
	typedFn.setWritable(false);
	addAttribute(outValue);
	
	MPointArray defaultPntArray;
	MFnPointArrayData pntArrayDataFn;
	pntArrayDataFn.create( defaultPntArray );
	
	adoplen = numFn.create( "dopLen", "dpl", MFnNumericData::kInt, 0 );
	numFn.setStorable(true);
	addAttribute(adoplen);
	
	MVectorArray defaultVecArray;
	MFnVectorArrayData vecArrayDataFn;
	vecArrayDataFn.create( defaultVecArray );
	adopPBuf = typedFn.create( "dopPBuf", "dpp",
											MFnData::kVectorArray,
											vecArrayDataFn.object(),
											&stat );
    typedFn.setStorable(true);
	addAttribute(adopPBuf);
	
	adopNBuf = typedFn.create( "dopNBuf", "dpn",
											MFnData::kVectorArray,
											vecArrayDataFn.object(),
											&stat );
    typedFn.setStorable(true);
	addAttribute(adopNBuf);
	
	adopCBuf = typedFn.create( "dopCBuf", "dpc",
											MFnData::kVectorArray,
											vecArrayDataFn.object(),
											&stat );
    typedFn.setStorable(true);
	addAttribute(adopCBuf);
	
	MFnMatrixAttribute matAttr;
	aininstspace = matAttr.create("instanceSpace", "sinst", MFnMatrixAttribute::kDouble);
	matAttr.setStorable(false);
	matAttr.setWritable(true);
	matAttr.setConnectable(true);
    matAttr.setArray(true);
    matAttr.setDisconnectBehavior(MFnAttribute::kDelete);
	addAttribute( aininstspace );
	
	attributeAffects(aradiusMult, outValue);
	attributeAffects(adoplen, outValue);
	attributeAffects(adopPBuf, outValue);
	attributeAffects(adrawColorR, outValue);
	attributeAffects(adrawColorG, outValue);
	attributeAffects(adrawColorB, outValue);
	attributeAffects(adrawDopSizeX, outValue);
	attributeAffects(adrawDopSizeY, outValue);
	attributeAffects(adrawDopSizeZ, outValue);
	attributeAffects(avoxactive, outValue);
	attributeAffects(avoxvisible, outValue);
	attributeAffects(adrawVoxTag, outValue);
	return MS::kSuccess;
}

MStatus ExampViz::connectionMade ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	if(plug == outValue)
		AHelper::Info<MString>("connect", plug.name());
	return MPxLocatorNode::connectionMade (plug, otherPlug, asSrc );
}

MStatus ExampViz::connectionBroken ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	if(plug == outValue)
		AHelper::Info<MString>("disconnect", plug.name());
	return MPxLocatorNode::connectionMade (plug, otherPlug, asSrc );
}

void ExampViz::voxelize3(sdb::VectorArray<cvx::Triangle> * tri,
							const BoundingBox & bbox)
{
	ExampVox::voxelize3(tri, bbox);
	
	MFnNumericData bbFn;
	MObject bbData = bbFn.create(MFnNumericData::k3Float);
	
	bbFn.setData(bbox.data()[0], bbox.data()[1], bbox.data()[2]);
	MPlug bbmnPlug(thisMObject(), abboxminv);
	bbmnPlug.setValue(bbData);
	
	bbFn.setData(bbox.data()[3], bbox.data()[4], bbox.data()[5]);
	MPlug bbmxPlug(thisMObject(), abboxmaxv);
	bbmxPlug.setValue(bbData);
	
	const int n = pntBufLength();
	MPlug dopLenPlug(thisMObject(), adoplen);
	dopLenPlug.setInt(n);
	if(n < 1) {
		return;
	}
	
	MVectorArray dopp; dopp.setLength(n);
	MVectorArray dopn; dopn.setLength(n);
	MVectorArray dopc; dopc.setLength(n);
	const Vector3F * ps = pntPositionR();
	const Vector3F * ns = pntNormalR();
	const Vector3F * cs = pntColorR();
	for(int i=0; i<n; ++i) {
		dopp[i] = MVector(ps[i].x, ps[i].y, ps[i].z);
		dopn[i] = MVector(ns[i].x, ns[i].y, ns[i].z);
		dopc[i] = MVector(cs[i].x, cs[i].y, cs[i].z);
	}
	
	MFnVectorArrayData vecFn;
	MObject opnt = vecFn.create(dopp);
	MPlug doppPlug(thisMObject(), adopPBuf);
	doppPlug.setValue(opnt);
	
	MObject onor = vecFn.create(dopn);
	MPlug dopnPlug(thisMObject(), adopNBuf);
	dopnPlug.setValue(onor);
	
	MObject ocol = vecFn.create(dopc);
	MPlug dopcPlug(thisMObject(), adopCBuf);
	dopcPlug.setValue(ocol);
	
	AHelper::Info<int>("reduced draw n point ", pntBufLength() );
	
}

void ExampViz::voxelize4(sdb::VectorArray<cvx::Triangle> * tri,
							const BoundingBox & bbox)
{
	voxelize3(tri, bbox);
	
	const int & np = pntBufLength();
	AHelper::Info<int>("voxelize4 n point ", np );
	const Vector3F * pr = pntPositionR();
	const Vector3F * nr = pntNormalR();
	const float sz0 = bbox.getLongestDistance() * .399f;
	
	PosNml smp;	
	VGDTyp valGrd;
	valGrd.fillBox(bbox, sz0 );
	for(int i=0;i<np;++i) {
		smp._pos = pr[i];
		smp._nml = nr[i];
	    valGrd.insertValueAtLevel(3, pr[i], smp);
	}
	valGrd.finishInsert();
	DrawGrid2::createPointBased<VGDTyp, PosNml> (&valGrd, 3);
	
	float ucol[3] = {.23f, .81f, .45f};
	setUniformColor(ucol);
	
}

void ExampViz::voxelize3(const aphid::DenseMatrix<float> & pnts,
						const MIntArray & triangleVertices,
						const aphid::BoundingBox & bbox)
{
	const int nind = triangleVertices.length();
	const int ntri = nind / 3;
	const float * vps = pnts.column(0);
						
	sdb::VectorArray<cvx::Triangle> tris;
	
	for(int i=0;i<ntri;++i) {
		aphid::cvx::Triangle atri;
		
		for(int j=0;j<3;++j) {
			const float * ci = &vps[triangleVertices[i*3+j] * 3];
			Vector3F fp(ci[0], ci[1], ci[2]);
			atri.setP(fp, j);
		}
		
		tris.insert(atri);
	}
	
	voxelize4(&tris, bbox);
}

void ExampViz::updateGeomBox(MObject & node)
{
	MPlug radiusMultPlug(node, aradiusMult);
	float radiusScal = radiusMultPlug.asFloat();
	setGeomSizeMult(radiusScal);
	
	BoundingBox bb;
	
	MObject bbmn;
	MPlug bbmnPlug(node, abboxminv);
	bbmnPlug.getValue(bbmn);
	MFnNumericData bbmnFn(bbmn);
	bbmnFn.getData3Float(bb.m_data[0], bb.m_data[1], bb.m_data[2]);
	
	MObject bbmx;
	MPlug bbmxPlug(node, abboxmaxv);
	bbmxPlug.getValue(bbmx);
	MFnNumericData bbmxFn(bbmx);
	bbmxFn.getData3Float(bb.m_data[3], bb.m_data[4], bb.m_data[5]);
	
	setGeomBox(&bb);
				
}

bool ExampViz::loadTriangles(MDataBlock & data)
{
	int n = data.inputValue(adoplen).asInt();
	if(n < 1) {
		AHelper::Info<int>(" ExampViz error zero n triangle", n);
		return false;
	}
	
	MDataHandle pntH = data.inputValue(adopPBuf);
	MFnVectorArrayData pntFn(pntH.data());
	MVectorArray pnts = pntFn.array();
	
	if(pnts.length() < n) {
		AHelper::Info<unsigned>(" ExampViz error wrong triangle position length", pnts.length() );
		return false;
	}
	
	MDataHandle nmlH = data.inputValue(adopNBuf);
	MFnVectorArrayData nmlFn(nmlH.data());
	MVectorArray nmls = nmlFn.array();
	
	if(nmls.length() < n) {
		AHelper::Info<unsigned>(" ExampViz error wrong triangle normal length", nmls.length() );
		return false;
	}
	
	MDataHandle colH = data.inputValue(adopCBuf);
	MFnVectorArrayData colFn(colH.data());
	MVectorArray cols = colFn.array();
	
	AHelper::Info<int>("col len", cols.length() );
	
	if(cols.length() != n) {
		fillDefaultCol(cols, n);
	}
	
	buildDrawBuf(n, pnts, nmls, cols);
	return true;
}

bool ExampViz::loadTriangles(MObject & node)
{
	MPlug doplenPlug(node, adoplen);
	int n = doplenPlug.asInt();
	if(n<1) {
		return false;
	}
	
	MPlug doppPlug(node, adopPBuf);
	MObject doppObj;
	doppPlug.getValue(doppObj);
	
	MFnVectorArrayData pntFn(doppObj);
	MVectorArray pnts = pntFn.array();
	
	unsigned np = pnts.length();
	if(np < n ) {
		AHelper::Info<unsigned>(" ExampViz error wrong triangle position length", np );
		return false;
	}
	
	MPlug dopnPlug(node, adopNBuf);
	MObject dopnObj;
	dopnPlug.getValue(dopnObj);
	
	MFnVectorArrayData norFn(dopnObj);
	MVectorArray nmls = norFn.array();
	
	unsigned nn = nmls.length();
	if(nn < n ) {
		AHelper::Info<unsigned>(" ExampViz error wrong triangle normal length", nn );
		return false;
	}
	
	MPlug dopcPlug(node, adopCBuf);
	MObject dopcObj;
	dopcPlug.getValue(dopcObj);
	
	MFnVectorArrayData colFn(dopcObj);
	MVectorArray cols = colFn.array();
	
	unsigned nc = cols.length();
	if(nc != n ) {
		fillDefaultCol(cols, n);
	}
	
	buildDrawBuf(n, pnts, nmls, cols);
	return true;
}

void ExampViz::fillDefaultCol(MVectorArray & cols,
					int n)
{
	cols.setLength(n);
	float * diffCol = diffuseMaterialColV();
	const MVector c(diffCol[0], diffCol[1], diffCol[2]);
	for(int i=0;i<n;++i) {
		cols[i] = c;
	}
	std::cout<<"\n ExampViz::fillDefaultCol "<<n
			<<" rgb "<<diffCol[0]<<","<<diffCol[1]<<","<<diffCol[2];
}

void ExampViz::buildDrawBuf(int n,
				const MVectorArray & pnts,
				const MVectorArray & nmls,
				const MVectorArray & cols)
{
	setPointDrawBufLen(n);
	
	Vector3F * ps = pntPositionR();
	Vector3F * ns = pntNormalR();
	Vector3F * cs = pntColorR();
	for(int i=0; i<n; ++i) {
		ps[i].set(pnts[i].x, pnts[i].y, pnts[i].z);
		ns[i].set(nmls[i].x, nmls[i].y, nmls[i].z);
		cs[i].set(cols[i].x, cols[i].y, cols[i].z);
	}
	
	AHelper::Info<unsigned>(" ExampViz load n point", n );
	
	const BoundingBox & bbox = geomBox();
	buildBounding8Dop(bbox);
	
	const float sz0 = bbox.getLongestDistance() * .399f;
	
	PosNml smp;
	VGDTyp valGrd;
	valGrd.fillBox(bbox, sz0 );
	for(int i=0;i<n;++i) {
		smp._pos = Vector3F(ps[i].x, ps[i].y, ps[i].z);
		smp._nml = Vector3F(ns[i].x, ns[i].y, ns[i].z);
	    valGrd.insertValueAtLevel(3, Vector3F(pnts[i].x, pnts[i].y, pnts[i].z),
								smp);
	}
	valGrd.finishInsert();
	DrawGrid2::createPointBased<VGDTyp, PosNml> (&valGrd, 3);
	
	setUniformDopColor(diffuseMaterialColV() );
	setUniformColor(diffuseMaterialColV() );
}

void ExampViz::updateGridUniformColor(const float * col)
{
	bool stat = false;
	if(m_preDiffCol[0] != col[0]) {
		m_preDiffCol[0] = col[0];
		stat = true;
	}
	
	if(m_preDiffCol[1] != col[1]) {
		m_preDiffCol[1] = col[1];
		stat = true;
	}
	
	if(m_preDiffCol[2] != col[2]) {
		m_preDiffCol[2] = col[2];
		stat = true;
	}
	
	if(!stat) {
		return;
	}
	
	setUniformColor(col);
}
//:~