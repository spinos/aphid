/*
 *  measures.h
 *  
 *
 *  Created by jian zhang on 2/15/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef SDF_MEASURES_H
#define SDF_MEASURES_H

#include <math/ANoise3.h>
#include <kd/IntersectEngine.h>
#include <kd/ClosestToPointEngine.h>

struct TransformAprox {

	float _u[3];
	float _scaling;
	float _offset[3];
	
	void setU(const float* p) {
		_u[0] = p[0] + _offset[0];
		_u[1] = p[1] + _offset[1];
		_u[2] = p[2] + _offset[2];
		_u[0] *= _scaling;
		_u[1] *= _scaling;
		_u[2] *= _scaling;
	}
	
	void setU(const Vector3F& p) {
		setU((const float*)&p);
	}
	
};

struct MeasureSphere {

	float measureAt(const float& x, const float& y, const float& z) {
		
		float cx = x * 1.1f + .1f;
		float cy = y * .9f + .3f;
		float cz = z * .7f + .8f;
		float r = sqrt(cx * cx + cy * cy + cz * cz);
		return r - 1.1f;
	
	}
};

struct MeasureNoise {

	float measureAt(const float & x, const float & y, const float & z) const
	{
		const Vector3F at(x, 1.03f, z);
		const Vector3F orp(-.5421f, -.7534f, -.386f);
		return y - ANoise3::Fbm((const float *)&at,
											(const float *)&orp,
											.7f,
											4,
											1.8f,
											.5f);
	}
};

struct KdMeasure {

	KdEngine _engine;
	ClosestToPointTestResult _ctx;
	KdNTree<PosSample, aphid::KdNNode<4> >* _tree;
	Vector3F _u;
	
	Vector3F _offset;
	float _scaling;
	
	float measureAt(const float& x, const float& y, const float& z) {

/// to world		
		_u.x = x * _scaling;
		_u.y = y * _scaling;
		_u.z = z * _scaling;
		_u += _offset;
	
		_ctx.reset(_u, 1e8f, true);
		_engine.closestToPoint<PosSample>(_tree, &_ctx);
		if(!_ctx._hasResult)
			return 0.f;
/// back to local, which is [-1,1]	
		return _ctx._distance / _scaling;
	}
	
};

#endif
