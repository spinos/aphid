/*
 *  TypedEntity.h
 *  
 *
 *  Created by jian zhang on 10/23/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
namespace aphid {

class TypedEntity {
public:
    enum Type {
		TUnknown = 0,
		TGenericMesh = 1,
        TTriangleMesh = 2,
		TPatchMesh = 3,
		TPolygonMesh = 4,
		TTetrahedronMesh = 5,
        TKdTree = 6,
        TTransform = 7,
		TJoint = 8,
		TTransformManipulator = 9,
		TDistantLight = 10,
		TPointLight = 11,
		TSquareLight = 12,
		TTexture = 13,
		TShader= 14,
		TCurve = 15,
		TBezierCurve = 16,
		TGeometryArray = 17,
		TPointCloud = 18,
		TOrientedBox = 19,
		TAttribute = 20
    };
    	
	virtual const Type type() const;
	
private:
};

}