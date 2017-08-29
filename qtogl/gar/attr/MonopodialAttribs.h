/*
 *  MonopodialAttribs.h
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_MONOPODIAL_ATTRIBS_H
#define GAR_MONOPODIAL_ATTRIBS_H

#include "PieceAttrib.h"

class MonopodialAttribs : public PieceAttrib {
	
	aphid::ATriangleMesh* m_cylinder;
	
public:
	MonopodialAttribs();
	
	virtual bool hasGeom() const;
	virtual int numGeomVariations() const;
	virtual aphid::ATriangleMesh* selectGeom(gar::SelectProfile* prof) const;
	virtual bool update();
	virtual bool isGeomStem() const;
	virtual bool isGeomBranchingUnit() const;
	virtual gar::BranchingUnitType getBranchingUnitType() const;
	virtual bool selectBud(gar::SelectBudContext* ctx) const;
	virtual void estimateExclusionRadius(float& minRadius);
	
private:
	bool selectTerminalBud(gar::SelectBudContext* ctx) const;
	bool selectLateralBud(gar::SelectBudContext* ctx) const;
	bool selectTerminalFoliage(gar::SelectBudContext* ctx) const;
	bool selectLateralFoliage(gar::SelectBudContext* ctx) const;
	
};

#endif
