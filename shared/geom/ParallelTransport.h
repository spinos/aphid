/*
 *  ParallelTransport.h
 *  
 *  minimal rotation
 *
 *  Created by jian zhang on 1/8/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_PBD_PARALLEL_TRANSPORT_H
#define APH_PBD_PARALLEL_TRANSPORT_H

namespace aphid {

class Vector3F;
class Matrix33F;

class ParallelTransport {

public:
	ParallelTransport();

	static void CurvatureBinormal(Vector3F& dst,
			const Vector3F& e0, const Vector3F& e1);
	static void ExtractSinAndCos(float& sinPhi, float& cosPhi,
			const float& kdk);
/// rotate u0 by e0 to e1	
	static void Rotate(Vector3F& u0, 
					const Vector3F& e0, const Vector3F& e1);
/// [T,N,B]
	static void FirstFrame(Matrix33F& frm, 
					const Vector3F& e0, const Vector3F& refv);
	static void RotateFrame(Matrix33F& frm, 
					const Vector3F& e0, const Vector3F& e1);
	static Vector3F FrameUp(const Matrix33F& frm);

};

}

#endif
