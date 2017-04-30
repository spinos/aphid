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
	
	void setLevel(int level);
	void setPatch(float* cvs, int* vertex, char* boundary, int* valence);
	void dice();
	void runTest();
	
private:
	int _level;
	int* _patch_set;
	char* _boundary;
	int* _cage_connection;
	float* _cage_vertices;
	
	int* _bent_connection;
	float* _bent_vertices;
};