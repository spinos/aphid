/*
 *  GeoElement.h
 *  convexHull
 *
 *  Created by jian zhang on 9/5/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
class GeoElement {
public:
	GeoElement();
	virtual ~GeoElement();
	
	virtual void clear();
	
	void setVisibility(char val);
	char isVisible() const;
	
	void setData(char *data);
	char *getData();
	
	void setMarked(char val);
	char isMarked() const;
	
	virtual void setIndex(int val);
	int getIndex() const;
	
	void setNext(GeoElement * another);
	GeoElement* getNext() const;
	
	GeoElement *next;
	char *m_data;
	int index;
	char visibility;
	char marked;
};