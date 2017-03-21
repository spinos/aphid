/*
 *  LandBlock.cpp
 *  
 *  a single piece of land
 *
 *  Created by jian zhang on 3/9/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */
 
#include "LandBlock.h"
#include <ttg/TetraMeshBuilder.h>
#include <ttg/GlobalElevation.h>
#include <ttg/GenericTetraGrid.h>
#include <ttg/HeightBccGrid.h>
#include <ttg/TetrahedronDistanceField.h>
#include <ttg/TetraGridTriangulation.h>
#include <geom/ATriangleMesh.h>
#include <img/HeightField.h>
#include <img/ImageSensor.h>

namespace aphid {

namespace ttg {

LandBlock::LandBlock(sdb::Entity * parent) : sdb::Entity(parent)
{
	const BoundingBox bx = buildBox();
	m_bccg = new BccTyp;
	m_bccg->fillBox(bx, 512.f);
	
	m_tetg = new TetGridTyp;
	m_field = new FieldTyp;
	m_mesher = new MesherTyp;
	m_frontMesh = new ATriangleMesh;
	m_heightField = new img::HeightField;
	m_level = 4;
}

LandBlock::~LandBlock()
{
	delete m_bccg;
	delete m_tetg;
	delete m_field;
	delete m_mesher;
	delete m_frontMesh;
	delete m_heightField;
}

void LandBlock::rebuild()
{
	const BoundingBox bx = buildBox();
	m_bccg->fillBox(bx, 512.f);
	processHeightField();
	triangulate();
}

void LandBlock::processHeightField()
{
	std::vector<int> fieldInds;
	getTouchedHeightFields(fieldInds);

	Array3<float> sig = img::HeightField::InitialValueAtLevel(9);
	
/// todo fusion
	if(fieldInds.size() > 0) {
		senseHeightField(sig, GlobalElevation::GetHeightField(fieldInds[0]));
	}
	
	m_heightField->create(sig);
	m_heightField->setRange(2048.f);
	m_heightField->verbose();

	sdb::AdaptiveGridDivideProfle subdprof;
	subdprof.setLevels(0, m_level);
	subdprof.setToDivideAllChild(true);
	subdprof.setOffset(0.f);
	
	m_bccg->subdivide<img::HeightField >(*m_heightField, subdprof);

	m_bccg->build();
	
	ttg::TetraMeshBuilder teter;
    teter.buildMesh(m_tetg, m_bccg);
	
	m_mesher->setGridField(m_tetg, m_field);
	
	img::BoxSampleProfile<float> sampler;
	sampler._channel = 0;
	sampler._defaultValue = 0.5f;
	
	m_heightField->getSampleProfile(&sampler, heightSampleFilterSize() );
		
	for(int i=0;i<m_field->numNodes();++i) {
		DistanceNode & d = m_field->nodes()[i];
		
		toUV(sampler._uCoord, sampler._vCoord, d.pos);
		d.val = d.pos.y - m_heightField->sampleHeight(&sampler);
		//d.val = elevation->sample(d.pos);
	}
	
	m_field->mergeShortEdges<img::HeightField, img::BoxSampleProfile<float> >(*m_heightField, sampler);
	
}

void LandBlock::toUV(float & u, float & v, const Vector3F & p) const
{ 
	u = (p.x + 1024.f) / 2048.f;
	v = (p.z + 1024.f) / 2048.f;
}

float LandBlock::heightSampleFilterSize() const
{ return 1.5f * m_bccg->levelCellSize(m_level) / m_heightField->sampleSize(); }

void LandBlock::triangulate()
{
	m_mesher->triangulate();
	m_mesher->dumpFrontTriangleMesh(m_frontMesh);
	m_frontMesh->calculateVertexNormals();
}

const LandBlock::TetGridTyp * LandBlock::grid() const
{ return m_tetg; }

const LandBlock::FieldTyp * LandBlock::field() const
{ return m_field; }

const ATriangleMesh * LandBlock::frontMesh() const
{ return m_frontMesh; }

void LandBlock::getTouchedHeightFields(std::vector<int> & inds) const
{
	const int n = GlobalElevation::NumHeightFields();
	if(n<1) {
		return;
	}
	const BoundingBox bx = buildBox();

	for(int i=0;i<n;++i) {
		const img::HeightField & fld = GlobalElevation::GetHeightField(i);
		if(fld.intersect(bx) ) {
			inds.push_back(i);
		}
	}
	
}

void LandBlock::senseHeightField(Array3<float> & sigY,
							const img::HeightField & fld) const
{
	img::ImageSensor<img::HeightField> sensor(Vector2F(-1024.f, -1024.f),
		Vector2F(1024.f, -1024.f), sigY.numCols(),
		Vector2F(-1024.f, 1024.f), sigY.numRows(),
		Vector2F(0.f, 0.f), Vector2F(1.f, 1.f) );
	sensor.verbose();
	
	sensor.sense(&sigY, 0, fld );
}

BoundingBox LandBlock::buildBox() const
{ return BoundingBox(-1023.99f, -1023.99f, -1023.99f, 1023.99f, 1023.99f, 1023.99f); }

}

}
