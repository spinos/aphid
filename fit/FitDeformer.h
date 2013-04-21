/*
 *  FitDeformer.h
 *  fit
 *
 *  Created by jian zhang on 4/21/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "AnchorDeformer.h"
#include <KdTree.h>
class FitDeformer : public AnchorDeformer {
public:
	FitDeformer();
	virtual ~FitDeformer();
	
	void setTarget(KdTree * tree);
	
	void fit();
private:
	KdTree * m_tree;
};