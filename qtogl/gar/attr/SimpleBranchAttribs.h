/*
 *  SimpleBranchAttribs.h
 *  
 *  synthesize from a stem and many leaves
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_SIMPLE_BRANCH_ATTRIBS_H
#define GAR_SIMPLE_BRANCH_ATTRIBS_H

#include "PieceAttrib.h"
#include <syn/BranchSynthesis.h>

namespace aphid {
class Matrix44F;
class SplineMap1D;
}

class SimpleBranchAttribs : public PieceAttrib, public gar::BranchSynthesis {
	
    PieceAttrib* m_inStemAttr;
	PieceAttrib* m_inLeafAttr;
	int m_instId;
	
	static int sNumInstances;
	
public:
	SimpleBranchAttribs();
	virtual ~SimpleBranchAttribs();
	
	virtual bool update();
/// multi instance of different settings
	virtual int attribInstanceId() const;
/// recv input attr
	virtual void connectTo(PieceAttrib* another, const std::string& portName);
/// when x < 1024 select stem geom x
/// when x >= 1024 select leaf geom x>>10
	virtual aphid::ATriangleMesh* selectGeom(gar::SelectProfile* prof) const;
	
	virtual bool isSynthesized() const;
	virtual int numSynthesizedGroups() const;
	virtual gar::SynthesisGroup* selectSynthesisGroup(gar::SelectProfile* prof) const;
	virtual bool canConnectToViaPort(const PieceAttrib* another, const std::string& portName) const;
	
private:
    bool connectToStem(PieceAttrib* another);
	bool connectToLeaf(PieceAttrib* another);
	aphid::ATriangleMesh* selectStemGeom(gar::SelectProfile* prof) const;
	aphid::ATriangleMesh* selectLeafGeom(gar::SelectProfile* prof) const;
	
};

#endif