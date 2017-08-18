/*
 *  DirectionalBendAttribs.h
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_DIRECTIONAL_BEND_ATTRIBS_H
#define GAR_DIRECTIONAL_BEND_ATTRIBS_H

#include "PieceAttrib.h"
#include <geom/PackTexcoord.h>

namespace aphid {
class DirectionalBendDeformer;

namespace smp {
class GeodesicSphere;
}

}

class DirectionalBendAttribs : public PieceAttrib, public aphid::PackTexcoord {

    PieceAttrib* m_inAttr;
	aphid::ATriangleMesh* m_inGeom;
	aphid::ATriangleMesh* m_outGeom[36];
	int m_instId;
	float m_exclR;
	float m_geomHeight;
	aphid::DirectionalBendDeformer* m_dfm;
	aphid::smp::GeodesicSphere* m_samples;
	static int sNumInstances;
	
public:
	DirectionalBendAttribs();
	
	void setInputGeom(aphid::ATriangleMesh* x);
	
	virtual bool hasGeom() const;
	virtual int numGeomVariations() const;
	virtual aphid::ATriangleMesh* selectGeom(gar::SelectProfile* prof) const;
	virtual bool update();
/// multi instance of different settings
	virtual int attribInstanceId() const;
/// recv input geom
	virtual void connectTo(PieceAttrib* another, const std::string& portName);
	virtual bool isGeomStem() const;
	virtual bool isGeomLeaf() const;
	virtual bool canConnectToViaPort(const PieceAttrib* another, const std::string& portName) const;
	
private:
    
};

#endif