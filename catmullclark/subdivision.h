/*
 *  subdivision.h
 *  qtbullet
 *
 *  Created by jian zhang on 7/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
 
class Subdivision {
public:
	Subdivision ();
	virtual ~Subdivision () {}
	
	virtual void draw();
	
private:
	int* _patch_set;
	int* _caga_connection;
	float* _cage_vertices;
	
	int* _bent_connection;
	float* _bent_vertices;
	
	int* _l2_connection;
	float* _l2_vertices;
	
	int* _l3_connection;
	float* _l3_vertices;
	
	int* _l4_connection;
	float* _l4_vertices;
	
	int* _l5_connection;
	float* _l5_vertices;
};