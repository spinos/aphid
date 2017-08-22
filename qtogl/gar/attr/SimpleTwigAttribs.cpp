/*
 *  SimpleTwigAttribs.cpp
 *  
 *  synthesize from a stem and many leaves
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "SimpleTwigAttribs.h"
#include <geom/ATriangleMesh.h>
#include <math/miscfuncs.h>
#include <math/SplineMap1D.h>
#include <gar_common.h>
#include "SynthesisGroup.h"
#include <geom/CylinderMesh.h>
#include <geom/SegmentNormals.h>

using namespace aphid;

int SimpleTwigAttribs::sNumInstances = 0;

SimpleTwigAttribs::SimpleTwigAttribs() : PieceAttrib(gar::gtSimpleTwig),
m_inStemAttr(NULL),
m_inLeafAttr(NULL)
{
    m_instId = sNumInstances;
	sNumInstances++;
	
	addFloatAttrib(gar::nPetioleAngle, .7f, 0.f, 1.5f);
	
	std::vector<int> phyllotaxyFields;
	phyllotaxyFields.push_back(gar::phOpposite);
	phyllotaxyFields.push_back(gar::phAlternate);
	phyllotaxyFields.push_back(gar::phDecussate);
	phyllotaxyFields.push_back(gar::phWhorled);
	
	addEnumAttrib(gar::nLeafPlacement, phyllotaxyFields);
	addIntAttrib(gar::nWhorlCount, 4, 3, 8);
	addFloatAttrib(gar::nGrowBegin, .3f, 0.f, .9f);
	addSplineAttrib(gar::nAgingVar);
	addSplineAttrib(gar::nSizeVariation);
	addSplineAttrib(gar::nFoldVar);
	addSplineAttrib(gar::nNoiseVariation);
	
	m_morph._sizingSpline = new SplineMap1D;
	m_morph._foldingSpline = new SplineMap1D;
	m_morph._noiseSpline = new SplineMap1D;
	m_morph._agingSpline = new SplineMap1D;
	
	gar::SplineAttrib* aas = (gar::SplineAttrib*)findAttrib(gar::nAgingVar);
	aas->setSplineValue(1.f, 0.f);
	aas->setSplineCv0(.4f, 1.f);
	aas->setSplineCv1(.6f, 0.f);
	
	gar::SplineAttrib* afs = (gar::SplineAttrib*)findAttrib(gar::nFoldVar);
	afs->setSplineValue(0.f, .5f);
	afs->setSplineCv0(.4f, .0f);
	afs->setSplineCv1(.6f, .5f);
	
}

SimpleTwigAttribs::~SimpleTwigAttribs()
{
	delete m_morph._sizingSpline;
	delete m_morph._foldingSpline;
	delete m_morph._noiseSpline;
	delete m_morph._agingSpline;
}

bool SimpleTwigAttribs::update()
{    
    if(!m_inStemAttr)
        return false;
	if(!m_inLeafAttr)
        return false;
		
	clearSynths();
	
	SplineMap1D* ags = m_morph._agingSpline;
	gar::SplineAttrib* aags = (gar::SplineAttrib*)findAttrib(gar::nAgingVar);
	updateSplineValues(ags, aags);
	
	SplineMap1D* ls = m_morph._sizingSpline;
	gar::SplineAttrib* als = (gar::SplineAttrib*)findAttrib(gar::nSizeVariation);
	updateSplineValues(ls, als);
	
	SplineMap1D* fs = m_morph._foldingSpline;
	gar::SplineAttrib* afs = (gar::SplineAttrib*)findAttrib(gar::nFoldVar);
	updateSplineValues(fs, afs);
	
	SplineMap1D* ns = m_morph._noiseSpline;
	gar::SplineAttrib* ans = (gar::SplineAttrib*)findAttrib(gar::nNoiseVariation);
	updateSplineValues(ns, ans);
	
	findAttrib(gar::nPetioleAngle)->getValue(m_morph._petioleAngle);
	findAttrib(gar::nLeafPlacement)->getValue(m_morph._leafPlacement);
	findAttrib(gar::nWhorlCount)->getValue(m_morph._whorlCount);
	findAttrib(gar::nGrowBegin)->getValue(m_morph._nodeBegin);
	
	gar::SelectProfile selprof;
	
/// for each stem variation
	const int ng = m_inStemAttr->numGeomVariations();
	for(int i=0;i<ng;++i) {
		selprof._index = i;
		ATriangleMesh* inGeom = m_inStemAttr->selectGeom(&selprof);
		if(!inGeom)
			return false;
			
		gar::SynthesisGroup * gi = addSynthesisGroup();
/// instance of stem
		gi->addInstance(i, Matrix44F::IdentityMatrix);
		gi->setExclusionRadius(selprof._exclR);
		synthsizeAGroup(gi, inGeom);
	}
    
	return true;
}

int SimpleTwigAttribs::attribInstanceId() const
{ return m_instId; }

bool SimpleTwigAttribs::connectToStem(PieceAttrib* another)
{
	if(!CanBeTwigStem(another->glyphType() ) )
		return false;
	
	m_inStemAttr = another;
	return true;
}

bool SimpleTwigAttribs::connectToLeaf(PieceAttrib* another)
{
	if(!CanBeTwigLeaf(another->glyphType() ) )
		return false;
		
	m_inLeafAttr = another;
	return true;
}

bool SimpleTwigAttribs::CanBeTwigStem(int x)
{
	const int gg = gar::ToGroupType(x);
	return (gg == gar::ggStem
		|| gg == gar::ggVariant);
}

bool SimpleTwigAttribs::CanBeTwigLeaf(int x)
{
	const int gg = gar::ToGroupType(x);
	return (gg == gar::ggSprite
		|| gg == gar::ggVariant);
}

bool SimpleTwigAttribs::canConnectToViaPort(const PieceAttrib* another, const std::string& portName) const
{
	if(portName == "inStem") 
		return another->isGeomStem();
	return another->isGeomLeaf();
}

void SimpleTwigAttribs::connectTo(PieceAttrib* another, const std::string& portName)
{
	bool stat = false;
	if(portName == "inStem") 
		stat = connectToStem(another);
	else
		stat = connectToLeaf(another);
		
    if(!stat) {
        std::cout<<"\n ERROR SimpleTwigAttribs cannot connect input attr ";
        return;
    }
    
    update();
}

ATriangleMesh* SimpleTwigAttribs::selectStemGeom(gar::SelectProfile* prof) const
{
    if(!m_inStemAttr)
			return NULL;
#if 0	
		std::cout<<"\n stem attr"<<m_inStemAttr->glyphType()
		<<" instance"<<m_inStemAttr->attribInstanceId()
		<<" geom"<<prof->_index;
		std::cout.flush();
#endif		
	prof->_geomInd = (gar::GlyphTypeToGeomIdGroup(m_inStemAttr->glyphType() ) 
	                    | (m_inStemAttr->attribInstanceId() << 10) 
	                    | prof->_index);
	
	return m_inStemAttr->selectGeom(prof);
}

ATriangleMesh* SimpleTwigAttribs::selectLeafGeom(gar::SelectProfile* prof) const
{
    if(!m_inLeafAttr)
		return NULL;
/// real index	
	prof->_index = (prof->_index - 1)>>10;
#if 0	
		std::cout<<"\n leaf attr"<<m_inLeafAttr->glyphType()
		<<" instance"<<m_inLeafAttr->attribInstanceId()
		<<" geom"<<prof->_index;
		std::cout.flush();
#endif	
	prof->_geomInd = (gar::GlyphTypeToGeomIdGroup(m_inLeafAttr->glyphType() ) 
	                    | (m_inLeafAttr->attribInstanceId() << 10) 
	                    | prof->_index);
	return m_inLeafAttr->selectGeom(prof);
}

ATriangleMesh* SimpleTwigAttribs::selectGeom(gar::SelectProfile* prof) const
{
/// always by index
	if(prof->_index < 1024) {
	    return selectStemGeom(prof);
	}
	
	return selectLeafGeom(prof);
}

bool SimpleTwigAttribs::isSynthesized() const
{ return true; }

int SimpleTwigAttribs::numSynthesizedGroups() const
{ return synthsisGroups().size(); }

gar::SynthesisGroup* SimpleTwigAttribs::selectSynthesisGroup(gar::SelectProfile* prof) const
{
	if(prof->_condition != gar::slIndex)
		prof->_index = rand() % numSynthesizedGroups();
	
	prof->_exclR = synthsisGroups()[prof->_index]->exclusionRadius();	
	return synthsisGroups()[prof->_index]; 
}

void SimpleTwigAttribs::synthsizeAGroup(gar::SynthesisGroup* grp,
					const ATriangleMesh* stemGeom)
{
	const int nvrow = CylinderMesh::GetNumVerticesPerRow(stemGeom);
	const int nv = stemGeom->numPoints();
	const int nrow = nv / nvrow;
	
	m_morph._deltaNodeParam = 1.f / (float)nrow;
	m_morph._nodeParam = 0.f;
	
	Vector3F* rowmeans = new Vector3F[nrow];
	rowmeans[0].setZero();
	for(int i=1;i<nrow;++i) {
		rowmeans[i] = CylinderMesh::GetRowMean(stemGeom, i, nvrow);
	}
	
	SegmentNormals segnml(nrow - 1);
	
	Vector3F p0p1 = rowmeans[1] - rowmeans[0];
	segnml.calculateFirstNormal(p0p1, Vector3F::XAxis );
	
	for(int i=1;i<nrow-1;++i) {
		
		const Vector3F& p0 = rowmeans[i - 1];
		const Vector3F& p1 = rowmeans[i];
		const Vector3F& p2 = rowmeans[i+1];
		p0p1 = p1 - p0;
		Vector3F p1p2 = p2 - p1;
		Vector3F p1pm02 = (p0 + p2) * 0.5f - p1;
		segnml.calculateNormal(i, p0p1, p1p2, p1pm02 );
		
	}
	
	m_morph._phyllotaxyAngle = 0.f;
	m_morph._nodeInd = 0;

	Matrix44F segmat;
	Matrix44F petmat;
	Matrix44F phymat;
/// for each segment
	for(int i=1;i<nrow-1;++i) {
		const Vector3F& p0 = rowmeans[i - 1];
		const Vector3F& p1 = rowmeans[i];
/// z points to grow direction
		p0p1 = p1 - p0;
		p0p1.normalize();
		Vector3F vup = segnml.getNormal(i);
		Vector3F vside = vup.cross(p0p1);
		vside.normalize();
		vup = p0p1.cross(vside);
		vup.normalize();
		
		float& noiseWeight = m_morph._nodeNoiseWeight;
		noiseWeight = m_morph._noiseSpline->interpolate(m_morph._nodeParam);
		
		float d = m_morph._sizingSpline->interpolate(m_morph._nodeParam);
		if(noiseWeight > 1e-3f) 
			d += RandomFn11() * 0.03f * noiseWeight;
		if(d < .07f) 
			d = .07f;
			
		m_morph._nodeScaling = d;
		
		segmat.setOrientations(vside * d, vup * d, p0p1 * d);
		segmat.setTranslation(p1);
		
		d = m_morph._petioleAngle + m_morph._foldingSpline->interpolate(m_morph._nodeParam);
		
		if(noiseWeight > 1e-3f) {
			d += RandomFn11() * 0.13f * noiseWeight;
		}
		
		if(d > 1.41f) 
			d = 1.41f;
		
		Quaternion petq(d, Vector3F::XAxis );
		Matrix33F mrot(petq);
		petmat.setRotation(mrot);
		
		if(m_morph._nodeParam > m_morph._nodeBegin)
			processPhyllotaxy(grp, petmat, segmat);
		
		m_morph._nodeParam += m_morph._deltaNodeParam;
	}
	
	delete[] rowmeans;
}

void SimpleTwigAttribs::processPhyllotaxy(gar::SynthesisGroup* grp,
			const Matrix44F& petmat,
			const Matrix44F& segmat)
{
/// rotate relative to last segment
/// and between petiole
	float segAng = 0.f, petAng = 0.f;
/// # leaf per node
	int nleaf = 1;
	switch(m_morph._leafPlacement) { 
		case gar::phOpposite:
			petAng = PIF;
			nleaf = 2;
		break;
		case gar::phAlternate:
			segAng = PIF;
		break;
		case gar::phDecussate:
			segAng = HALFPIF;
			petAng = PIF;
			nleaf = 2;
		break;
		case gar::phWhorled:
			petAng = TWOPIF / (float)m_morph._whorlCount;
			nleaf = m_morph._whorlCount;
			segAng = petAng * .29f;
		break;
		default :
		;
	
	}
	
	const float& noiseWeight = m_morph._nodeNoiseWeight;
	if(noiseWeight > 1e-3f) {
		petAng *= 1.f + RandomFn11() * 0.17f * noiseWeight;
	}
	
	if(m_morph._leafPlacement == gar::phOpposite
		|| m_morph._leafPlacement == gar::phWhorled) {
/// skip odd nodes
		if(m_morph._nodeInd & 1) {
			nleaf = 0;
		}
	}
	
	gar::SelectProfile selprof;
	selprof._condition = gar::slAge;
	selprof._age = m_morph._agingSpline->interpolate(m_morph._nodeParam);
	
	Matrix44F phymat;
	const Vector3F ppos = segmat.getTranslation();
	const Vector3F vrot = segmat.getFront().normal();
	for(int i=0;i<nleaf;++i) {
		
		Quaternion phyq(m_morph._phyllotaxyAngle + petAng * i, vrot );
		Matrix33F mrot(phyq);
		phymat.setRotation(mrot);
		
/// combine rotation then restore translation
		Matrix44F instmat = petmat * segmat * phymat;
		instmat.setTranslation(ppos);
	
		m_inLeafAttr->selectGeom(&selprof);
		grp->addInstance((selprof._index+1) << 10, instmat);
		grp->adjustExclusionRadius(.73f * selprof._height * m_morph._nodeScaling );
	}
	
	m_morph._phyllotaxyAngle += segAng;
	m_morph._nodeInd++;
}
