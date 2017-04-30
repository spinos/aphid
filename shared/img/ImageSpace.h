/*
 *  ImageSpace.h
 *  
 *	space transform between destination and source image
 *
 *  Created by jian zhang on 3/18/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_IMG_IMAGE_SPACE_H
#define APH_IMG_IMAGE_SPACE_H

#include <math/Matrix44F.h>

namespace aphid {

namespace img {

struct ImageSpace {
	
/// 1.0 / num sample in destination
	float _sampleSpacing;
/// 1.0 / num sample in source
	float _sourceSpacing;
	Matrix44F _worldToUVMatrix;
	
/// from world point to texcoord	
	void toUV(float & u, float & v,
				const Vector2F & p)
	{
		Vector2F q = _worldToUVMatrix.transform(p);
		u = q.x;
		v = q.y;
	}
	
};

}

}
#endif