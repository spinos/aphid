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
}

namespace gar {
class SynthesisGroup;

struct SelectProfile {

	SelectCondition _condition;
	int _index;
	float _exclR;
	float _height;
	float _age;
	
	SelectProfile() {
		_condition = slIndex;
		_index = 0;
		_exclR = 1.f;
		_height = 1.f;
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
/// set upstream
	virtual void connectTo(PieceAttrib* another, const std::string& portName);
/// width / height, for uv packing
	virtual float texcoordBlockAspectRatio() const;
/// for synthesized
	virtual bool isSynthesized() const;
	virtual int numSynthesizedGroups() const;
	virtual gar::SynthesisGroup* selectSynthesisGroup(gar::SelectProfile* prof) const;
	virtual bool isGeomStem() const;
	virtual bool isGeomLeaf() const;
	
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
		
	void addSplineAttrib(gar::AttribName anm);
		
	void addStringAttrib(gar::AttribName anm,
		const std::string& val,
		const bool& asFileName = false);
		
	void addEnumAttrib(gar::AttribName anm,
		const std::vector<int>& fields);
	
	void updateSplineValues(aphid::SplineMap1D* ls, gar::SplineAttrib* als);
	
private:
};

#endif
