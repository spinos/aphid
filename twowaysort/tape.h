/*
 *  tape.h
 *  twowaysort
 *
 *  Created by jian zhang on 2/6/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <fstream>
class ATape
{
public:
	ATape();
	virtual ~ATape();
	void openIn(const char *fileName);
	void openOut(const char *fileName);
	void close();
	void finalize();
	char getF(float &val);
	void setF(const float &val);
	int readBlock(int loc, int size, char *data);
	void writeBlock(int size, char *data);
	
	int readLocation(int size) const;
	int fileEnd() const;
	void setReadLimit(int min, int max);
	char exhausted() const;
	void print(int runl);
	std::string fileName() const;
	
private:
	std::fstream _stream;
	std::string _fileName;
	int _loc, _bufferLoc;
	int _readMin, _readMax;
	int _end;
	char *_buffer;
};