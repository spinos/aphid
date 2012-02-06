/*
 *  tape.cpp
 *  twowaysort
 *
 *  Created by jian zhang on 2/6/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "tape.h"
#include <iostream>

#define BUFFERSIZE 524288

ATape::ATape() 
{	
	_buffer = new char[BUFFERSIZE];
}

ATape::~ATape() 
{
	delete[] _buffer;
}

void ATape::openIn(const char *fileName)
{
	_fileName = fileName;
	_loc = 0;
	_stream.open(_fileName.c_str(), std::fstream::in | std::fstream::binary);
	
	_stream.seekg (0, std::ios::end);
	_end = _stream.tellg();
}

void ATape::openOut(const char *fileName)
{
	_fileName = fileName;
	_loc = 0;
	_bufferLoc = 0;
	_stream.open(_fileName.c_str(), std::fstream::out | std::fstream::binary);
}

void ATape::close()
{
	_stream.close();
}



char ATape::getF(float &val)
{
	if(_loc >= _readMax || _loc >= _end)
		return 0;
		
	if(_loc % BUFFERSIZE == 0) {
		_stream.seekg (_loc, std::ios::beg);
		int readSize = _end - _loc;
		if(readSize > BUFFERSIZE)
			readSize = BUFFERSIZE;
		_stream.read((char*)_buffer, readSize);
	}
	
	float *p = (float *)_buffer;
	val = p[(_loc % BUFFERSIZE) / 4];
	_loc += 4;
	return 1;
}

void ATape::setF(const float &val)
{
	if(_bufferLoc == BUFFERSIZE) {
		_stream.write((char*)_buffer, BUFFERSIZE);
		_bufferLoc = 0;
	}
	float *p = (float *)_buffer;
	p[_bufferLoc / 4] = val;
	_loc += 4;
	_bufferLoc += 4;
}

int ATape::readBlock(int loc, int size, char *data)
{
	if(loc >= _end)
		return 0;
		
	_stream.seekg (loc, std::ios::beg);
	int readSize = _end - loc;
	if(readSize > size)
		readSize = size;
		
	_stream.read(data, readSize);
	return readSize;
}

void ATape::writeBlock(int size, char *data)
{
	_stream.write(data, size);
}

void ATape::finalize()
{
	if(_bufferLoc > 0) {
		std::cout<<"write last "<<_fileName<<" "<<_bufferLoc<<"\n";
		_stream.write((char*)_buffer, _bufferLoc);
	}
}

int ATape::readLocation(int size) const
{
	return _loc/size;
}

void ATape::setReadLimit(int min, int max)
{
	_readMin = min;
	_readMax = max;
}

int ATape::fileEnd() const
{
	return _end;
}

char  ATape::exhausted() const
{
	return (_loc >= _end || _readMin >= _end || _loc >= _readMax);
}

void ATape::print(int runl)
{
	_loc = 0;
	_stream.open(_fileName.c_str(), std::fstream::in | std::fstream::binary);
	
	_stream.seekg (0, std::ios::end);
	_end = _stream.tellg();

	std::cout<<"start printing "<<_fileName<<"\n";
	
	for(int i=0; i<_end/4; i++)
	{
		if(i%runl == 0)
			std::cout<<" | ";
		float v;
		_stream.seekg (i*4, std::ios::beg);
		_stream.read((char*)&v, 4);
		std::cout<<" "<<v;
	}
	_stream.close();
	std::cout<<"\neof\n";
}

std::string ATape::fileName() const
{
	return _fileName;
}
