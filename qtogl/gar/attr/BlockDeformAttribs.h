/*
 *  BlockDeformAttribs.h
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_BLOCK_DEFORM_ATTRIBS_H
#define GAR_BLOCK_DEFORM_ATTRIBS_H

#include "PieceAttrib.h"
#include <geom/PackTexcoord.h>

namespace aphid {
class BlockDeformer;
class Vector3F;
class Matrix44F;
}

class BlockDeformAttribs : public PieceAttrib, public aphid::PackTexcoord {

    PieceAttrib* m_inAttr;
	aphid::ATriangleMesh* m_inGeom;
	aphid::ATriangleMesh* m_outGeom[48];
/// 8 mat per variation
	float m_blkMat[48][128];
	int m_instId;
	float m_exclR;
	float m_geomHeight;
	aphid::BlockDeformer* m_dfm;
	
	static int sNumInstances;
	
public:
	BlockDeformAttribs();
	
	virtual bool hasGeom() const;
	virtual int numGeomVariations() const;
	virtual aphid::ATriangleMesh* selectGeom(gar::SelectProfile* prof) const;
	virtual bool update();
/// multi instance of different settings
	virtual int attribInstanceId() const;
/// recv input geom
	virtual void connectTo(PieceAttrib* another, const std::string& portName);
/// clear input geom
	virtual void disconnectFrom(PieceAttrib* another, const std::string& portName);
/// depend on in attr
	virtual bool isGeomStem() const;
	virtual bool isGeomLeaf() const;
	virtual bool isGeomBranchingUnit() const;
	virtual bool canConnectToViaPort(const PieceAttrib* another, const std::string& portName) const;
	virtual gar::BranchingUnitType getBranchingUnitType() const;
	virtual bool selectBud(gar::SelectBudContext* ctx) const;
	virtual void estimateExclusionRadius(float& minRadius);
	
private:
    bool selectTerminalBud(gar::SelectBudContext* ctx) const;
	bool selectLateralBud(gar::SelectBudContext* ctx) const;
	bool selectAllLateralBud(gar::SelectBudContext* ctx) const;
	bool selectCloseToUpLateralBud(gar::SelectBudContext* ctx) const;

/// last block up of i-th variation
	aphid::Vector3F variationDirection(int i) const;
	int variationCloseToUp(gar::SelectProfile* prof) const;
/// rotate around local tm y-axis to face z-axis up
	void rotateToUp(aphid::Matrix44F& tm, const aphid::Vector3F& relup) const;
/// first rotate y-axis around world_y to perpendicular to up
/// then z-axis rotate around new y to face up 
	void rotateToUp2(aphid::Matrix44F& tm, const aphid::Vector3F& relup,
				const aphid::Vector3F& worldy) const;
	
};

#endif