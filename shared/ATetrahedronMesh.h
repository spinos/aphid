#ifndef ATETRAHEDRONMESH_H
#define ATETRAHEDRONMESH_H

/*
 *  ATetrahedronMesh.h
 *  aphid
 *
 *  Created by jian zhang on 4/25/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include "AGenericMesh.h"
#include "BaseSampler.h"

class ATetrahedronMesh : public AGenericMesh {
public:
	ATetrahedronMesh();
	virtual ~ATetrahedronMesh();
	
	virtual const Type type() const;
	virtual const unsigned numComponents() const;
	virtual const BoundingBox calculateBBox(unsigned icomponent) const;
	virtual void closestToPoint(unsigned icomponent, ClosestToPointTestResult * result);
	virtual bool intersectBox(unsigned icomponent, const BoundingBox & box);
    const unsigned numTetrahedrons() const;
	
	void create(unsigned np, unsigned nt);
	unsigned * tetrahedronIndices(unsigned idx) const;
    float calculateVolume() const;
    
    const float volume() const;
    void setVolume(float x);
    
    TetrahedronSampler * sampler();
	
	virtual std::string verbosestr() const;
protected:
	
private:
    TetrahedronSampler m_sampler;
	float m_volume;
};
#endif        //  #ifndef ATETRAHEDRONMESH_H
