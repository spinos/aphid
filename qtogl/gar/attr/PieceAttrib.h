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
}

class PieceAttrib {

typedef std::vector<gar::Attrib* > AttribArrayTyp;
	AttribArrayTyp m_collAttrs;
	std::string m_glyphName;
	
public:	
	PieceAttrib();
	virtual ~PieceAttrib();
	
	const std::string& glyphName() const;
	
	int numAttribs() const;
	gar::Attrib* getAttrib(const int& i);
	const gar::Attrib* getAttrib(const int& i) const;
	gar::Attrib* findAttrib(gar::AttribName anm);
	const gar::Attrib* findAttrib(gar::AttribName anm) const;
	
	virtual bool hasGeom() const;
	virtual int numGeomVariations() const;
	virtual aphid::ATriangleMesh* selectGeom(int x, float& exclR) const;
	virtual bool update();
/// multi instance of different settings
	virtual int attribInstanceId() const;
	
protected:
	void addIntAttrib(gar::AttribName anm,
		const int& val, 
		const int& minVal,
		const int& maxVal);
		
	void addFloatAttrib(gar::AttribName anm,
		const float& val, 
		const float& minVal = 0.f,
		const float& maxVal = 1.f);
		
	void addSplineAttrib(gar::AttribName anm);
		
	void addStringAttrib(gar::AttribName anm,
		const std::string& val,
		const bool& asFileName = false);
	
private:
};

#endif
