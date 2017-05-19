/*
 *  MVegExample.h
 *  proxyPaint
 *
 *  Created by jian zhang on 5/17/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef RHI_M_VEG_EXAMPLE_H
#define RHI_M_VEG_EXAMPLE_H

#include "GardenExamp.h"
#include <maya/MVectorArray.h>
#include <maya/MIntArray.h>
#include <maya/MPlug.h>

namespace aphid {

class MVegExample : public GardenExamp, public CachedExampParam {

public:
	MVegExample();
	virtual ~MVegExample();
	
protected:
	int loadGroupBBox(const MPlug & boxPlug);
	int loadInstance(const MPlug & drangePlug,
					const MPlug & dindPlug,
					const MPlug & dtmPlug);
	void loadPoints(const MPlug & rangePlug,
					const MPlug & pncPlug);
	void loadHull(const MPlug & rangePlug,
					const MPlug & pnPlug);
	void loadVoxel(const MPlug & rangePlug,
					const MPlug & pncPlug);
	
	void loadExmpInstance(CompoundExamp * exmp,
					const int & ibegin, const int & iend,
					const MIntArray & dind,
					const MVectorArray & dtm);
	void loadExmpPoints(CompoundExamp * exmp,
					const int & ibegin, const int & iend,
					const MVectorArray & dpnc);
	void loadExmpHull(CompoundExamp * exmp,
					const int & ibegin, const int & iend,
					const MVectorArray & dpnc);
	void loadExmpVoxel(CompoundExamp * exmp,
					const int & ibegin, const int & iend,
					const MVectorArray & dpnc);
	
	void saveInstanceTo(MVectorArray & dtm, MIntArray & dind,
						CompoundExamp * exmp) const;
	
	void saveExmpPoints(MVectorArray & dst, CompoundExamp * exmp) const;
	void saveExmpHull(MVectorArray & dst, CompoundExamp * exmp) const;
	void saveExmpVoxel(MVectorArray & dst, CompoundExamp * exmp) const;
	int saveGroupBBox(MPlug & boxPlug);
	int saveInstance(MPlug & drangePlug, 
					MPlug & dindPlug, 
					MPlug & dtmPlug);
	int savePoints(MPlug & drangePlug, 
					MPlug & dpntPlug);
	int saveHull(MPlug & drangePlug, 
					MPlug & dpntPlug);
	int saveVoxel(MPlug & drangePlug, 
					MPlug & dpntPlug);
	void drawExampPoints(int idx);
	void drawExampHull(int idx);
	void updateAllDop();
	void updateAllDetailDrawType();
	
	void buildAllExmpVoxel();
	
private:
	void updateExampDop(CompoundExamp * exmp,
				const float * col,
			const float * sz);
	
};

}

#endif