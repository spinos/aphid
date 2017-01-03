/*
 *  FeatherGeomParam.cpp
 *  cinchona
 *
 *  Created by jian zhang on 1/3/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "FeatherGeomParam.h"
#include <math/Vector3F.h>
#include <math/linspace.h>

using namespace aphid;

FeatherGeomParam::FeatherGeomParam() 
{
	_numTotalF = 0;
	for(int i=0;i<4;++i) {
		_numFPerSeg[i] = 0;
		_xsPerSeg[i] = NULL;
		_psPerSeg[i] = NULL;
	}
}

FeatherGeomParam::~FeatherGeomParam()
{
	for(int i=0;i<4;++i) {
		if(_xsPerSeg[i]) delete[] _xsPerSeg[i];
		if(_psPerSeg[i]) delete[] _psPerSeg[i];
	}
}

void FeatherGeomParam::set(const int * nps)
{
	_changed = false;
	_numTotalF = 0;
	for(int i=0;i<4;++i) {
		const int nf = nps[i];
		if(nf != _numFPerSeg[i]) {
			_changed = true;
			_numFPerSeg[i] = nf;
			 
			if(_xsPerSeg[i]) {
				delete[] _xsPerSeg[i];
			}
			_xsPerSeg[i] = new float[nf];
			
/// at center of each segment
			linspace_center<float>(_xsPerSeg[i], 0.f, 1.f, nf);
			
			if(_psPerSeg[i]) {
				delete[] _psPerSeg[i];
			}
			_psPerSeg[i] = new Vector3F[nf];
		}
		_numTotalF += nf;
		
	}
}

bool FeatherGeomParam::isChanged() const
{ return _changed; }
