/*
 *  twoWayMerge.h
 *  twowaysort
 *
 *  Created by jian zhang on 2/7/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

class ATape;
class TwoWayMerge
{
public:
	TwoWayMerge();
	char doIt(const char *filename, int dataSize);
	char validate();
private:
	void split(const char *filename, int dataSize);
	void merge();
	
	ATape tapeDst1;
	ATape tapeDst2;
	ATape tapeSrc1;
	ATape tapeSrc2;
	std::string _resultFileName;
	int _dataLength;
	int _dataSize;
};