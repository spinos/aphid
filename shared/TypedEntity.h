/*
 *  TypedEntity.h
 *  
 *
 *  Created by jian zhang on 10/23/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

class TypedEntity {
public:
    enum TypeEntries {
		TUnknown = 0,
        TTriangleMesh = 1,
        TPatchMesh = 2,
		TKdTree = 3,
        TTransform = 4,
		TJoint = 5,
		TTransformManipulator = 6,
		TDistantLight = 7,
		TPointLight = 8,
		TSquareLight = 9
    };
    
	TypedEntity();
	void setEntityType(TypeEntries val);
	int entityType() const;
	
	bool isMesh() const;
	bool isTriangleMesh() const;
	bool isPatchMesh() const;
private:	
	int m_type;
};