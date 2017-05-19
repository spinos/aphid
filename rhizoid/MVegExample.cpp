/*
 *  MVegExample.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 5/17/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "MVegExample.h"
#include <CompoundExamp.h>
#include <mama/AttributeHelper.h>
#include <maya/MFnVectorArrayData.h>
#include <maya/MFnIntArrayData.h>
#include <gl_heads.h>

namespace aphid {

MVegExample::MVegExample()
{}

MVegExample::~MVegExample()
{}

int MVegExample::saveGroupBBox(MPlug & boxPlug)
{
	const int nexmp = numExamples();
	MVectorArray dbox; dbox.setLength(nexmp * 2);
	for(int i=0;i<nexmp;++i) {
		CompoundExamp * cxmp = getCompoundExample(i);
		const BoundingBox & gbx = cxmp->geomBox();
		dbox[i*2] = MVector(gbx.getMin(0), gbx.getMin(1), gbx.getMin(2) );
		dbox[i*2 + 1] = MVector(gbx.getMax(0), gbx.getMax(1), gbx.getMax(2) );
	}
	
	AttributeHelper::SaveArrayDataPlug<MVectorArray, MFnVectorArrayData > (dbox, boxPlug);
	std::cout<<"\n save n group "<<nexmp;
	return nexmp;
}

int MVegExample::saveInstance(MPlug & drangePlug, 
								MPlug & dindPlug, 
								MPlug & dtmPlug)
{
	const int nexmp = numExamples();
	MVectorArray dtm;
	MIntArray dind;
	MIntArray drange; drange.setLength(nexmp+1);
	int b = 0;
	for(int i=0;i<nexmp;++i) {
		CompoundExamp * cxmp = getCompoundExample(i);
		saveInstanceTo(dtm, dind, cxmp);
		drange[i] = b;
		const int c = cxmp->numInstances();
		b += c;
	}
	drange[nexmp] = b;
	
	AttributeHelper::SaveArrayDataPlug<MIntArray, MFnIntArrayData > (drange, drangePlug);
	
	AttributeHelper::SaveArrayDataPlug<MIntArray, MFnIntArrayData > (dind, dindPlug);
	
	AttributeHelper::SaveArrayDataPlug<MVectorArray, MFnVectorArrayData > (dtm, dtmPlug);
	std::cout<<"\n save n instance tm "<<b;
	return b;
}

int MVegExample::savePoints(MPlug & drangePlug, 
								MPlug & dpntPlug)
{
	MVectorArray dpnt;
	MIntArray drange;
	const int nexmp = numExamples();
	int c = 0;
	for(int i=0;i<nexmp;++i) {
		CompoundExamp * cxmp = getCompoundExample(i);
		saveExmpPoints(dpnt, cxmp);
		drange.append(c );
		c += cxmp->pntBufLength();
	}
	drange.append(c );
	
	AttributeHelper::SaveArrayDataPlug<MVectorArray, MFnVectorArrayData > (dpnt, dpntPlug);
	
	AttributeHelper::SaveArrayDataPlug<MIntArray, MFnIntArrayData > (drange, drangePlug);
	std::cout<<"\n save point draw nv "<<c;
	return c;
}

int MVegExample::saveHull(MPlug & drangePlug, MPlug & dpntPlug)
{
	MVectorArray dpnt;
	MIntArray drange;
	const int nexmp = numExamples();
	int c = 0;
	for(int i=0;i<nexmp;++i) {
		CompoundExamp * cxmp = getCompoundExample(i);
		saveExmpHull(dpnt, cxmp);
		drange.append(c );
		c += cxmp->dopBufLength();
	}
	drange.append(c );
	
	AttributeHelper::SaveArrayDataPlug<MIntArray, MFnIntArrayData > (drange, drangePlug);
	AttributeHelper::SaveArrayDataPlug<MVectorArray, MFnVectorArrayData > (dpnt, dpntPlug);
	std::cout<<"\n save hull draw nv "<<c;
	return c;
}

int MVegExample::saveVoxel(MPlug & drangePlug, MPlug & dpntPlug)
{
	MVectorArray dpnt;
	MIntArray drange;
	int c = 0;
	const int nexmp = numExamples();
	for(int i=0;i<nexmp;++i) {
		CompoundExamp * cxmp = getCompoundExample(i);
		saveExmpVoxel(dpnt, cxmp);
		drange.append(c );
		c += cxmp->grdBufLength();
	}
	drange.append(c );
	
	AttributeHelper::SaveArrayDataPlug<MVectorArray, MFnVectorArrayData > (dpnt, dpntPlug);
	
	AttributeHelper::SaveArrayDataPlug<MIntArray, MFnIntArrayData > (drange, drangePlug);
	std::cout<<"\n save voxel draw nv "<<c;
	return c;
}

void MVegExample::saveInstanceTo(MVectorArray & dtm, MIntArray & dind,
						CompoundExamp * exmp) const
{
	const int n = exmp->numInstances();
	for(int i=0;i<n;++i) {
		const InstanceD & ainst = exmp->getInstance(i);
		const float * tm = ainst._trans;
		dtm.append(MVector(tm[0], tm[1], tm[2]));
		dtm.append(MVector(tm[4], tm[5], tm[6]));
		dtm.append(MVector(tm[8], tm[9], tm[10]));
		dtm.append(MVector(tm[12], tm[13], tm[14]));
		dind.append(ainst._instanceId);
	}
}

void MVegExample::saveExmpPoints(MVectorArray & dst, CompoundExamp * exmp) const
{
	const int & np = exmp->pntBufLength();
	Vector3F * nr = exmp->pntNormalR();
	Vector3F * pr = exmp->pntPositionR();
	Vector3F * cr = exmp->pntColorR();
	for(int i=0;i<np;++i) {
		dst.append(MVector(pr[i].x, pr[i].y, pr[i].z) );
		dst.append(MVector(nr[i].x, nr[i].y, nr[i].z) );
		dst.append(MVector(cr[i].x, cr[i].y, cr[i].z) );
	}
}

void MVegExample::saveExmpHull(MVectorArray & dst, CompoundExamp * exmp) const
{
	const int & np = exmp->dopBufLength();
	float * nr = exmp->dopNormalR();
	float * pr = exmp->dopRefPositionR();
	for(int i=0;i<np;++i) {
		int i3 = i * 3;
		dst.append(MVector(pr[i3], pr[i3 + 1], pr[i3 + 2]) );
		dst.append(MVector(nr[i3], nr[i3 + 1], nr[i3 + 2]) );
	}
}

void MVegExample::saveExmpVoxel(MVectorArray & dst, CompoundExamp * exmp) const
{
	const int & np = exmp->grdBufLength();
	float * nr = exmp->grdNormalR();
	float * pr = exmp->grdPositionR();
	float * cr = exmp->grdColorR();
	for(int i=0;i<np;++i) {
		int i3 = i * 3;
		dst.append(MVector(pr[i3], pr[i3 + 1], pr[i3 + 2]) );
		dst.append(MVector(nr[i3], nr[i3 + 1], nr[i3 + 2]) );
		dst.append(MVector(cr[i3], cr[i3 + 1], cr[i3 + 2]) );
	}
}

void MVegExample::drawExampPoints(int idx)
{
	const ExampVox * ve = getExample(idx);
	glPushAttrib(GL_LIGHTING_BIT | GL_CURRENT_BIT);
	glDepthFunc(GL_LEQUAL);
	glDisable(GL_LIGHTING);
	glColor3f(1,1,1);
	
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	
	ve->drawPoints();
	
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	
	glPopAttrib();
}

void MVegExample::drawExampHull(int idx)
{
	const ExampVox * ve = getExample(idx);
	glPushAttrib(GL_LIGHTING_BIT | GL_CURRENT_BIT);
	glDepthFunc(GL_LEQUAL);
	glDisable(GL_LIGHTING);
	glColor3f(1,1,1);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	
	ve->drawASolidDop();
	
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glPopAttrib();
}

int MVegExample::loadGroupBBox(const MPlug & boxPlug)
{
	MVectorArray dbox;
	AttributeHelper::LoadArrayDataPlug<MVectorArray, MFnVectorArrayData > (dbox, boxPlug);
	const int nexmp = dbox.length() >> 1;
	if(nexmp < 1) {
		return nexmp;
	}
	
	BoundingBox bbox;
	for(int i=0;i<nexmp;++i) {
		int i2 = i<<1;
		const MVector & vlo = dbox[i2];
		const MVector & vhi = dbox[i2 + 1];
		
		CompoundExamp * cxmp = new CompoundExamp;
		bbox.setMin(vlo.x, vlo.y, vlo.z);
		bbox.setMax(vhi.x, vhi.y, vhi.z);
		cxmp->setGeomBox2(bbox);
		
		addAExample(cxmp);
	}
	std::cout<<"\n load n group "<<nexmp;
	return nexmp;
}

int MVegExample::loadInstance(const MPlug & drangePlug,
					const MPlug & dindPlug,
					const MPlug & dtmPlug)
{
	MIntArray drange;
	AttributeHelper::LoadArrayDataPlug<MIntArray, MFnIntArrayData > (drange, drangePlug);
	
	MIntArray dind;
	AttributeHelper::LoadArrayDataPlug<MIntArray, MFnIntArrayData > (dind, dindPlug);
	
	MVectorArray dtm;
	AttributeHelper::LoadArrayDataPlug<MVectorArray, MFnVectorArrayData > (dtm, dtmPlug);
	
	const int nexmp = numExamples();
	const int nrange = drange.length();
	int c = 0;
	for(int i=0;i<nexmp;++i) {
		if(i+1 >= nrange) {
			std::cout<<"\n load instance error out of range "<<i;
			break;
		}
		
		CompoundExamp * cxmp = getCompoundExample(i);
		loadExmpInstance(cxmp, drange[i], drange[i+1],
						dind, dtm);
		c += cxmp->numInstances();
		
	}
	std::cout<<"\n load n instance tm "<<c;
	return 1;
}

void MVegExample::loadExmpInstance(CompoundExamp * exmp,
					const int & ibegin, const int & iend,
					const MIntArray & dind,
					const MVectorArray & dtm)
{
	const int nind = dind.length();
	const int ntm = dtm.length()>>2;
	float tm[16];
	memset(tm, 0, 64);
	tm[15] = 1.f;
	int instanceId;
	for(int i=ibegin;i<iend;++i) {
		if(i>= nind || i>= ntm) {
			std::cout<<"\n instance id out of range "<<i;
			break;
		}
		instanceId = dind[i];
		
		int i4 = i<<2;
		for(int j=0;j<4;++j) {
			const MVector & vj = dtm[i4 + j];
			tm[j * 4] = vj.x;
			tm[j * 4 + 1] = vj.y;
			tm[j * 4 + 2] = vj.z;
		}
		exmp->addInstance(tm, instanceId);
	}
}

void MVegExample::loadPoints(const MPlug & rangePlug,
					const MPlug & pncPlug)
{
	MIntArray drange;
	AttributeHelper::LoadArrayDataPlug<MIntArray, MFnIntArrayData > (drange, rangePlug);
	
	MVectorArray dpnc;
	AttributeHelper::LoadArrayDataPlug<MVectorArray, MFnVectorArrayData > (dpnc, pncPlug);
	
	const int nexmp = numExamples();
	const int nrange = drange.length();
	int c = 0;
	for(int i=0;i<nexmp;++i) {
		if(i+1 >= nrange) {
			std::cout<<"\n load points error out of range "<<i;
			break;
		}
		
		CompoundExamp * cxmp = getCompoundExample(i);
		loadExmpPoints(cxmp, drange[i], drange[i+1],
						dpnc);
		c += cxmp->pntBufLength();
	}
	std::cout<<"\n load n group point "<<c;
	
}

void MVegExample::loadHull(const MPlug & rangePlug,
					const MPlug & pnPlug)
{
	MIntArray drange;
	AttributeHelper::LoadArrayDataPlug<MIntArray, MFnIntArrayData > (drange, rangePlug);
	
	MVectorArray dpnc;
	AttributeHelper::LoadArrayDataPlug<MVectorArray, MFnVectorArrayData > (dpnc, pnPlug);
	
	const int nexmp = numExamples();
	const int nrange = drange.length();
	const int npnc = dpnc.length();
	std::cout<<" nexmp "<<nexmp<<" nrange"<<nrange<<" npnc "<<npnc;
	int c = 0;
	for(int i=0;i<nexmp;++i) {
		if(i+1 >= nrange) {
			std::cout<<"\n load hull error out of range "<<i;
			break;
		}
		
		CompoundExamp * cxmp = getCompoundExample(i);
		loadExmpHull(cxmp, drange[i], drange[i+1],
						dpnc);
		c += cxmp->dopBufLength();
	}
	std::cout<<"\n load n group hull "<<c;
	
}

void MVegExample::loadVoxel(const MPlug & rangePlug,
					const MPlug & pncPlug)
{
	MIntArray drange;
	AttributeHelper::LoadArrayDataPlug<MIntArray, MFnIntArrayData > (drange, rangePlug);
	
	MVectorArray dpnc;
	AttributeHelper::LoadArrayDataPlug<MVectorArray, MFnVectorArrayData > (dpnc, pncPlug);
	
	const int nexmp = numExamples();
	const int nrange = drange.length();
	int c = 0;
	for(int i=0;i<nexmp;++i) {
		if(i+1 >= nrange) {
			std::cout<<"\n load voxel error out of range "<<i;
			break;
		}
		
		CompoundExamp * cxmp = getCompoundExample(i);
		loadExmpVoxel(cxmp, drange[i], drange[i+1],
						dpnc);
		c += cxmp->grdBufLength();
	}
	std::cout<<"\n load n group voxel "<<c;

}

void MVegExample::loadExmpPoints(CompoundExamp * exmp,
					const int & ibegin, const int & iend,
					const MVectorArray & dpnc)
{
	const int npnc = dpnc.length() / 3;
	const int np = iend - ibegin;
	exmp->setPointDrawBufLen(np);
	Vector3F * nr = exmp->pntNormalR();
	Vector3F * pr = exmp->pntPositionR();
	Vector3F * cr = exmp->pntColorR();
	for(int i=0;i<np;++i) {
		int j = ibegin + i;
		if(j >= npnc) {
			std::cout<<"\n load group points error out of range "<<j;
			break;
		}
		int j3 = j * 3;
		const MVector & pj = dpnc[j3];
		const MVector & nj = dpnc[j3 + 1];
		const MVector & cj = dpnc[j3 + 2];
		pr[i] = Vector3F(pj.x, pj.y, pj.z);
		nr[i] = Vector3F(nj.x, nj.y, nj.z);
		cr[i] = Vector3F(cj.x, cj.y, cj.z);
		
	}
}

void MVegExample::loadExmpHull(CompoundExamp * exmp,
					const int & ibegin, const int & iend,
					const MVectorArray & dpnc)
{
	const int npnc = dpnc.length() >> 1;
	const int np = iend - ibegin;
	exmp->setDopDrawBufLen(np);
	float * nr = exmp->dopNormalR();
	float * pr = exmp->dopRefPositionR();
	//float * cr = exmp->dopColorR();
	for(int i=0;i<np;++i) {
		int j = ibegin + i;
		if(j >= npnc) {
			std::cout<<"\n load group hull error out of range "<<ibegin<<" "<<iend;
			break;
		}
		int j2 = j * 2;
		const MVector & pj = dpnc[j2];
		const MVector & nj = dpnc[j2 + 1];
		
		int i3 = i * 3;
		pr[i3] = pj.x; pr[i3 + 1] = pj.y; pr[i3 + 2] = pj.z;
		nr[i3] = nj.x; nr[i3 + 1] = nj.y; nr[i3 + 2] = nj.z;
		
	}
	
	exmp->resizeDopPoints(Vector3F(1.f, 1.f, 1.f) );
	
}

void MVegExample::loadExmpVoxel(CompoundExamp * exmp,
					const int & ibegin, const int & iend,
					const MVectorArray & dpnc)
{
	const int npnc = dpnc.length() / 3;
	const int np = iend - ibegin;
	exmp->setGrdDrawBufLen(np);
	float * nr = exmp->grdNormalR();
	float * pr = exmp->grdPositionR();
	float * cr = exmp->grdColorR();
	for(int i=0;i<np;++i) {
		int j = ibegin + i;
		if(j >= npnc) {
			std::cout<<"\n load group voxel error out of range "<<j;
			break;
		}
		int j3 = j * 3;
		const MVector & pj = dpnc[j3];
		const MVector & nj = dpnc[j3 + 1];
		const MVector & cj = dpnc[j3 + 2];
		memcpy(&pr[i * 3], &pj, 12);
		memcpy(&nr[i * 3], &nj, 12);
		memcpy(&cr[i * 3], &cj, 12);
		
	}
}

void MVegExample::updateAllDop()
{
	const float * col = diffuseMaterialColor();
	bool stat = isDiffColChanged(col);
	const float * sz = dopSize();
	if(isDspSizeChanged(sz) ) {
		stat = true;
	}
	
	if(!stat) {
		return;
	}
	
	const int nexmp = numExamples();
	for(int i=0;i<nexmp;++i) {
		CompoundExamp * cxmp = getCompoundExample(i);
		updateExampDop(cxmp, col, sz);
	}
}

void MVegExample::updateExampDop(CompoundExamp * exmp,
			const float * col,
			const float * sz)
{
	exmp->setUniformDopColor(col);
	exmp->resizeDopPoints(sz);
}

void MVegExample::updateAllDetailDrawType()
{
	const short & dt = detailDrawType();
	if(!isDrawTypeChanged(dt) ) {
		return;
	}
	const int nexmp = numExamples();
	for(int i=0;i<nexmp;++i) {
		CompoundExamp * cxmp = getCompoundExample(i);
		cxmp->setDetailDrawType(dt);
	}
}

void MVegExample::buildAllExmpVoxel()
{
	const int nexmp = numExamples();
	for(int i=0;i<nexmp;++i) {
		ExampVox * xmp = getCompoundExample(i);
		const BoundingBox & bbx = xmp->geomBox();
		xmp->buildVoxel(bbx);
	}
}

}