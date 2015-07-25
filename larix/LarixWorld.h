/*
 *  LarixWorld.h
 *  larix
 *
 *  Created by jian zhang on 7/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
class APointCloud;
class LarixWorld {
public:
	LarixWorld();
	virtual ~LarixWorld();
	
	void setPointCloud(APointCloud * pc);
	APointCloud * pointCloud() const;
protected:

private:
	APointCloud * m_cloud;
};