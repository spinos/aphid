/*
 *  ColorSampler.h
 *  mama
 *
 *  sample color by texture
 *
 *  Created by jian zhang on 12/28/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_MAMA_COLOR_SAMPLER_H
#define APH_MAMA_COLOR_SAMPLER_H

#include <mama/MeshHelper.h>
#include <math/Vector3F.h>

namespace aphid {

class ExrImage;

class ColorSampler : public MeshHelper {

public:
	struct SampleProfile {
		Vector3F m_defaultColor;
		ExrImage * m_imageSampler;
		
	};
	
	ColorSampler();

	static void SampleMeshTrianglesInGroup(sdb::VectorArray<cvx::Triangle> & tris,
								BoundingBox & bbox,
							const MDagPath & groupPath);

protected:
	static bool SampleTriangles(sdb::VectorArray<cvx::Triangle> & tris,
						const int & iBegin, const int & iEnd,
						SampleProfile * profile);
						
	static void GetMeshDefaultColor(SampleProfile * profile,
						const MObject & node);
	static void GetMeshImageFileName(SampleProfile * profile,
						const MObject & node);
	
};

};

#endif