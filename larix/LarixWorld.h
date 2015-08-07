/*
 *  LarixWorld.h
 *  larix
 *
 *  Created by jian zhang on 7/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <AWorld.h>
class ATetrahedronMesh;
class APointCloud;
class AdaptiveField;
class H5FieldIn;
class TypedBuffer;

class LarixWorld : public AWorld {
public:
	LarixWorld();
	virtual ~LarixWorld();
    
    void setTetrahedronMesh(ATetrahedronMesh * m);
    ATetrahedronMesh * tetrahedronMesh() const;
	
	void setPointCloud(APointCloud * pc);
	APointCloud * pointCloud() const;
	
	void setField(AdaptiveField * g);
	AdaptiveField * field() const;

    bool hasSourceP() const;
    TypedBuffer * sourceP();

    int cacheRangeMin() const;
    int cacheRangeMax() const;
    void beginCache();
    int currentCacheFrame() const;
// override aworld
    virtual void progressFrame();
protected:
    bool checkSourceField();
    void readCacheRange();
private:
    ATetrahedronMesh * m_mesh;
	APointCloud * m_cloud;
	AdaptiveField * m_field;
	H5FieldIn * m_sourceFile;
    TypedBuffer * m_sourceP;
};