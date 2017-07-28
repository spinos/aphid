/*
 *  SegmentNormals.h
 *
 *      |
 *  p0 --- p1
 *            \ /
 *             \
 *              p2
 *  up direction per segment
 *  first seg base on p01 and given ref
 *  next segs up close to last seg
 *  assuming angle change between segs less than 90 deg
 *
 *  Created by jian zhang on 7/29/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_GEOM_SEGMENT_NORMALS_H
#define APH_GEOM_SEGMENT_NORMALS_H

namespace aphid {

class Vector3F;

class SegmentNormals {

	Vector3F* m_ups;
	
public:
	SegmentNormals(int nseg);
	~SegmentNormals();
/// i-th seg	
	const Vector3F& getNormal(int i) const;
/// first segment facing ref
	void calculateFirstNormal(const Vector3F& p0p1,
					const Vector3F& ref);
/// segment i>0
/// p1p2 _|_ last_normal : use last normal as ref
/// else: use p1_to_mid_point
	void calculateNormal(int i, const Vector3F& p0p1,
						const Vector3F& p1p2,
						const Vector3F& p1pm02);
};

}

#endif
