/*
 *  PieceAttrib.h
 *
 *  collection of attribs
 *  update geom after any attrib changes
 *
 *  Created by jian zhang on 8/5/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_PIECE_ATTRIB_H
#define GAR_PIECE_ATTRIB_H

#include "Attrib.h"
#include <vector>

namespace aphid {
class ATriangleMesh;
class SplineMap1D;
class Vector3F;
}

class GrowthSampleProfile;

namespace gar {
class SynthesisGroup;

struct SelectProfile {

	SelectCondition _condition;
/// selection
	int _index;
/// attr_type | attr_instance | geom_selection
	int _geomInd;
/// exclusion radius
	float _exclR;
/// geometry
	float _height;
/// selection
	float _age;
/// close to up reference
	float _upVec[3];
/// space of growth
	float _relMat[16];
			
	SelectProfile() {
		_condition = slIndex;
		_index = 0;
		_exclR = 1.f;
		_height = 1.f;
		_upVec[0] = 0.f;
		_upVec[1] = 1.f;
		_upVec[2] = 0.f;
	}
	
};

/// select max 8 buds
struct SelectBudContext {

	BudType _budType;
	SelectCondition _condition;
/// local transform of bud
	float _budTm[8][16];
/// bind ind of bud
	int _budBind[8];
	int _numSelect;
	int _variationIndex;
/// eliminate downward growth 
	float _upVec[3];
	float _upLimit;
/// space of growth
	float _relMat[16];
/// ascend angle and age varing
	float _ascending;
	float _ascendVaring;
/// angle between leaf stalk and stem
	float _axil;
	
	SelectBudContext() {
		_budType = bdTerminal;
		_condition = slCloseToUp;
		_numSelect = 1;
		_variationIndex = 0;
		_upLimit = 0.f;
		_upVec[0] = 0.f;
		_upVec[1] = 1.f;
		_upVec[2] = 0.f;
		_ascending = .2f;
		_ascendVaring = 0.f;
		_axil = 1.2f;
	}
	
};

}

class PieceAttrib {

typedef std::vector<gar::Attrib* > AttribArrayTyp;
	AttribArrayTyp m_collAttrs;
	std::string m_glyphName;
	int m_glyphType;
	
public:	
	PieceAttrib(int glyphType = 0);
	virtual ~PieceAttrib();
	
	const std::string& glyphName() const;
	const int& glyphType() const;
	
	int numAttribs() const;
	gar::Attrib* getAttrib(const int& i);
	const gar::Attrib* getAttrib(const int& i) const;
	gar::Attrib* findAttrib(gar::AttribName anm);
	const gar::Attrib* findAttrib(gar::AttribName anm) const;
	
	virtual bool hasGeom() const;
	virtual int numGeomVariations() const;
	virtual aphid::ATriangleMesh* selectGeom(gar::SelectProfile* prof) const;
	virtual bool update();
/// multi instance of different settings
	virtual int attribInstanceId() const;
/// check possible upstream
	virtual bool canConnectToViaPort(const PieceAttrib* another, const std::string& portName) const;
/// update upstream
	virtual void connectTo(PieceAttrib* another, const std::string& portName);
	virtual void disconnectFrom(PieceAttrib* another, const std::string& portName);
	virtual void postPortChange(const std::string& portName);
/// width / height, for uv packing
	virtual float texcoordBlockAspectRatio() const;
/// for synthesized
	virtual bool isSynthesized() const;
	virtual int numSynthesizedGroups() const;
	virtual gar::SynthesisGroup* selectSynthesisGroup(gar::SelectProfile* prof) const;
	virtual bool resynthesize();
	virtual bool isGeomStem() const;
	virtual bool isGeomLeaf() const;
	virtual bool isGeomBranchingUnit() const;
	virtual gar::BranchingUnitType getBranchingUnitType() const;
	virtual bool selectBud(gar::SelectBudContext* ctx) const;
	virtual bool isTwig() const;
	virtual bool isBranch() const;
	virtual void estimateExclusionRadius(float& minRadius);
/// to grow on
	virtual bool isGround() const;
	virtual void getGrowthProfile(GrowthSampleProfile* prof) const;
/// can be deformed by profiles
	virtual bool isGeomProfiled() const;
	
protected:
	void addIntAttrib(gar::AttribName anm,
		const int& val, 
		const int& minVal,
		const int& maxVal);
		
	void addFloatAttrib(gar::AttribName anm,
		const float& val, 
		const float& minVal = 0.f,
		const float& maxVal = 1.f);
		
	void addVector2Attrib(gar::AttribName anm,
		const float& val0, 
		const float& val1);
		
	void addInt2Attrib(gar::AttribName anm,
		const int& val0, 
		const int& val1);
		
	void addSplineAttrib(gar::AttribName anm);
		
	void addStringAttrib(gar::AttribName anm,
		const std::string& val,
		const bool& asFileName = false);
		
	void addEnumAttrib(gar::AttribName anm,
		const std::vector<int>& fields);
		
	void addActionAttrib(gar::AttribName anm,
		const std::string& imgname);
	
	void updateSplineValues(aphid::SplineMap1D* ls, gar::SplineAttrib* als);
	
	aphid::Vector3F getLocalUpRef(gar::SelectBudContext* ctx) const;
	
private:
};

#endif
