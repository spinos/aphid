/*
 *  WindTurbine.h
 *  rotor and stator of a wind turbine for wind properties
 *  and visualiztion
 *
 *  Created by jian zhang on 7/30/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_PBD_WIND_TURBINE_MESH_H
#define APH_PBD_WIND_TURBINE_MESH_H

#include <math/Matrix44F.h>

namespace aphid {

namespace pbd {

class WindTurbine {

	Matrix44F m_vizSpace;
	float m_windSpeed;
	float m_rotorAngle;
	
public:
	WindTurbine();
	virtual ~WindTurbine();
	
	Matrix44F* visualizeSpace();
	const Matrix44F* visualizeSpace() const;
	
	void setWindSpeed(float x);
	const float& windSpeed() const;
/// v_tau direction and speed, update viz space as well
    void setMeanWindVec(const Vector3F& vtau);
/// v_tau
/// +x facing wind direction
	Vector3F getMeanWindVec() const;
/// angular x of rotor
	const float& rotorAngle() const;
/// rotate
	void progress(float dt);
	
	static const int sStatorNumVertices;
	static const int sStatorNumTriangleIndices;
	static const int sStatorMeshTriangleIndices[];
	static const float sStatorMeshVertices[];
	static const float sStatorMeshNormals[];
	
	static const int sRotorNumVertices;
	static const int sRotorNumTriangleIndices;
	static const int sRotorMeshTriangleIndices[];
	static const float sRotorMeshVertices[];
	static const float sRotorMeshNormals[];
	
	static const int sBladeNumVertices;
	static const int sBladeNumTriangleIndices;
	static const int sBladeMeshTriangleIndices[];
	static const float sBladeMeshVertices[];
	static const float sBladeMeshNormals[];
	
protected:

private:

};

}
}
#endif
