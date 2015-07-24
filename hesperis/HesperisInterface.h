/*
 *  HesperisInterface.h
 *  larix
 *
 *  Created by jian zhang on 7/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <string>
class CurveGroup;
class GeometryArray;
class HesperisInterface {
public:
	HesperisInterface();
	virtual ~HesperisInterface();
	
	static bool CheckFileExists();
	static bool ReadCurveData(CurveGroup * data);
	static bool ReadTriangleData(GeometryArray * data);
	static bool ReadTetrahedronData(GeometryArray * data);
	
	static std::string FileName;
protected:

private:

};