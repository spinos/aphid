/*
 *  DeformationAnalysisDrawer.h
 *  lapl
 *
 *  Created by jian zhang on 3/24/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "KdTreeDrawer.h"
#include "DeformationAnalysis.h"

class DeformationAnalysisDrawer : public KdTreeDrawer {
public:
	DeformationAnalysisDrawer();
	virtual ~DeformationAnalysisDrawer();
	
	void visualize(DeformationAnalysis * analysis);
	
private:
	
};