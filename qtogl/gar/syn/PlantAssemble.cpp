/*
 *  PlantAssemble.cpp
 *  
 *
 *  Created by jian zhang on 8/29/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "PlantAssemble.h"
#include <PlantPiece.h>
#include <graphchart/GardenGlyph.h>
#include <attr/PieceAttrib.h>
#include <qt/GlyphPort.h>
#include <qt/GlyphConnection.h>
#include <Vegetation.h>
#include <VegetationPatch.h>
#include "GrowthSample.h"
#include "SynthesisGroup.h"
#include <gar_common.h>
#include <data/ground.h>
#include <data/grass.h>

using namespace aphid;

namespace gar {

PlantAssemble::PlantAssemble(Vegetation * vege) 
{ m_vege = vege; }

Vegetation* PlantAssemble::vegetationR()
{ return m_vege; }

void PlantAssemble::genSinglePlant()
{
	Vegetation* vege = vegetationR();
	vege->clearGeom();
	GardenGlyph * gnd = getGround();
	if(!gnd) {
		vege->setNumPatches(0);
		return;
	}
	
	estimateExclusionRadius(gnd);
	growOnGround(vege->patch(0), gnd);
	
	vege->setNumPatches(1);
	vege->voxelize();
}
	
void PlantAssemble::genMultiPlant()
{
	Vegetation* vege = vegetationR();
	vege->clearGeom();
	GardenGlyph * gnd = getGround();
	if(!gnd) {
		vege->setNumPatches(0);
		return;
	}
	
	const int n = vege->getMaxNumPatches();
	for(int i=0;i<n;++i) {
		std::cout<<"\n n patches to go "<<(n - i);
		growOnGround(vege->patch(i), gnd);
	}
	vege->setNumPatches(n);
	vege->rearrange();
	vege->voxelize();
}

GardenGlyph* PlantAssemble::getGround()
{ return NULL; }

GardenGlyph* PlantAssemble::checkGroundConnection(GardenGlyph* gnd)
{
	int nc = 0;
	foreach(QGraphicsItem *port_, gnd->childItems()) {
		if (port_->type() != GlyphPort::Type) {
			continue;
		}

		GlyphPort *port = (GlyphPort*) port_;
		if(!port->isOutgoing() ) {
			nc += port->numConnections();
		}
	}
	
	if(nc < 1)
		return NULL;
		
	return gnd;
}

void PlantAssemble::estimateExclusionRadius(GardenGlyph * gnd)
{
	float minR = 1e8f;
	foreach(QGraphicsItem *port_, gnd->childItems()) {
		if (port_->type() != GlyphPort::Type) {
			continue;
		}

		GlyphPort *port = (GlyphPort*) port_;
		if(!port->isOutgoing() ) {
			const int n = port->numConnections();
			for(int i=0;i<n;++i) {
				const GlyphConnection * conn = port->connection(i);
				const GlyphPort * srcpt = conn->port0();
				QGraphicsItem * top = srcpt->topLevelItem();
				GardenGlyph * gl = (GardenGlyph *)top;
				PieceAttrib* attr = gl->attrib();
				attr->estimateExclusionRadius(minR);
			}
		}
	}
	m_exclR = minR;
}

void PlantAssemble::growOnGround(VegetationPatch * vpatch, GardenGlyph * gnd)
{
	GrowthSampleProfile prof;
	
	prof._exclR = m_exclR;
	prof.m_tilt = vpatch->tilt();
	
	PieceAttrib * gndAttr = gnd->attrib();
	gndAttr->getGrowthProfile(&prof);
	
	GrowthSample gsmp;
	
	switch (gnd->glyphType()) {
		case gar::gtPot:
			gsmp.samplePot(prof);
			break;
		case gar::gtBush:
			gsmp.sampleBush(prof);
			break;
		default:
			break;
	}
	
	vpatch->clearPlants();

	const int& np = gsmp.numGrowthSamples();
	for(int i=0;i<np;++i) {
		PlantPiece* pl = new PlantPiece;
		assemblePlant(pl, gnd );
		Matrix44F tm = gsmp.getGrowSpace(i, prof);
		pl->setTransformMatrix(tm);
		vpatch->addPlant(pl);
	}
	
}

void PlantAssemble::assemblePlant(PlantPiece * pl, GardenGlyph * gl)
{
	foreach(QGraphicsItem *port_, gl->childItems()) {
		if (port_->type() != GlyphPort::Type) {
			continue;
		}

		GlyphPort *port = (GlyphPort*) port_;
		if(!port->isOutgoing() ) {
			addPiece(pl, port);
			
		}
	}
	pl->setExclRByChild();
	
}

void PlantAssemble::addPiece(PlantPiece * pl, const GlyphPort * pt)
{
	const int n = pt->numConnections();
	if(n < 1) {
		return;
	}
/// randomly select an input
	int r = rand() % n;
	const GlyphConnection * conn = pt->connection(r);
	const GlyphPort * srcpt = conn->port0();
	QGraphicsItem * top = srcpt->topLevelItem();
	if(top->type() != GardenGlyph::Type) {
		return;
	}
		
	GardenGlyph * gl = (GardenGlyph *)top;
	const int gg = gar::ToGroupType(gl->glyphType() );
		
	PlantPiece * pl1 = new PlantPiece(pl);
	switch (gg) {
		case gar::ggGrass:
		case gar::ggSprite:
		case gar::ggVariant:
		case gar::ggStem:
			addSinglePiece(pl1, gl);
		break;
		case gar::ggTwig:
			addTwigPiece(pl1, gl);
		break;
		case gar::ggBranch:
			addBranchPiece(pl1, gl);
		break;
		default:
		;
	}	
}

void PlantAssemble::addBranchPiece(PlantPiece * pl, GardenGlyph * gl)
{
	PieceAttrib* attr = gl->attrib();
	attr->resynthesize();
/// select first	
	SelectProfile selprof;
	SynthesisGroup* syng = attr->selectSynthesisGroup(&selprof);
	pl->setExclR(selprof._exclR);
	
	addSynthesisPiece(pl, attr, syng);
}

void PlantAssemble::addTwigPiece(PlantPiece * pl, GardenGlyph * gl)
{
	PieceAttrib* attr = gl->attrib();
/// select randomly
	SelectProfile selprof;
	selprof._condition = gar::slRandom;
	
	SynthesisGroup* syng = attr->selectSynthesisGroup(&selprof);
	pl->setExclR(selprof._exclR);

	addSynthesisPiece(pl, attr, syng);
}

void PlantAssemble::addSynthesisPiece(PlantPiece * pl, PieceAttrib* attr, 
							SynthesisGroup* syng)
{
	const int& ninst = syng->numInstances();
	
	SelectProfile selinst;	
	Matrix44F tm;
/// first as root piece
	syng->getInstance(selinst._index, tm, 0);
	ATriangleMesh * msh = attr->selectGeom(&selinst);
	const int& kgeom = selinst._geomInd;
	
	if(!m_vege->findGeom(kgeom)) {
		m_vege->addGeom(kgeom, msh);
	}
	
	int geomInd = m_vege->getGeomInd(msh);
	pl->setGeometry(msh, geomInd);
	
/// rest as child pieces
	for(int i=1;i<ninst;++i) {
		syng->getInstance(selinst._index, tm, i);
		
		ATriangleMesh * childMsh = attr->selectGeom(&selinst);
		const int& kchildGeom = selinst._geomInd;
	
		if(!m_vege->findGeom(kchildGeom)) {
			m_vege->addGeom(kchildGeom, childMsh);
		}
		
		int childGeomInd = m_vege->getGeomInd(childMsh);
		PlantPiece* childPiece = new PlantPiece(pl);
		
		childPiece->setGeometry(childMsh, childGeomInd);
	    childPiece->setExclR(selinst._exclR);
	    childPiece->setTransformMatrix(tm);
	    
	}
}

void PlantAssemble::addSinglePiece(PlantPiece * pl, GardenGlyph * gl)
{
	PieceAttrib* attr = gl->attrib();
/// select randomly
	SelectProfile selprof;
	selprof._condition = gar::slRandom;
	
	ATriangleMesh * msh = attr->selectGeom(&selprof);
/// node_type node_instance geom_ind	
	const int kgeom = gar::GlyphTypeToGeomIdGroup(gl->glyphType() ) | (gl->attribInstanceId() << 10) | selprof._index;
	
	if(!m_vege->findGeom(kgeom)) {
		m_vege->addGeom(kgeom, msh);
	}
	
	int geomInd = m_vege->getGeomInd(msh);
	pl->setGeometry(msh, geomInd);
	pl->setExclR(selprof._exclR);
}

}
