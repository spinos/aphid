/*
 *  twoWayMerge.cpp
 *  twowaysort
 *
 *  Created by jian zhang on 2/7/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include "tape.h"
#include "twoWayMerge.h"
#include <vector>
#include <algorithm>

#define BLOCKSIZE 524288

bool myCompareFunc(float i,float j) { return (i<j); }

void mySortFunc(float *array)
{
	std::vector <float> primes(array,  array + BLOCKSIZE);
		
		std::sort( primes.begin(), primes.end(), myCompareFunc);
		
		for(int i=0; i < BLOCKSIZE; i++) {
			array[i] = primes.at(i);
		}
}

void alternateTape(int &i) 
{
	if(i==0)
		i = 1;
	else
		i = 0;
}

void swap(float &a, float &b)
{
	float c = a;
	a = b;
	b = c;
}

void copyTape(ATape &src, ATape &dst, int length)
{
	for(int i = 0; i < length; i++) {		
		float v;
		if(src.getF(v))
			dst.setF(v);
		else
			break;
	}
}

void mergeTapes(ATape &src1, ATape &src2, ATape &dest, int runLength, int runStart)
{
#if defined DBG	
	std::cout<< "merge run start " << runStart/4 << " end " << runStart/4 + runLength << "\n";
#endif
	src1.setReadLimit(runStart, runStart + runLength * 4);
	src2.setReadLimit(runStart, runStart + runLength * 4);
	
	if(src2.exhausted()) {
#if defined DBG
		std::cout<<"no data on src2! copy src1 to dst\n";
#endif
		copyTape(src1, dest, runLength);
		return;
	}

	float lo, hi;
	src1.getF(lo);
	src2.getF(hi);
	
	int activeTape = 0;
	if(!myCompareFunc(lo, hi)) {
		swap(lo, hi);
		activeTape = 1;
	}
	dest.setF(lo);

	for(int i = 1; i < runLength * 2; i++) {
	
		if(activeTape == 0) {
			if(src1.exhausted()) {
#if defined DBG
				std::cout<<"src1 exhausted! copy src2 to dst\n";
#endif
				dest.setF(hi);
				copyTape(src2, dest, runLength);
				return;
			}
			src1.getF(lo);
		}
		else {
			if(src2.exhausted()) {
#if defined DBG
				std::cout<<"src2 exhausted! copy src1 to dst\n";
#endif
				dest.setF(hi);
				copyTape(src1, dest, runLength);
				return;
			}

			src2.getF(lo);
		}
		
		if(!myCompareFunc(lo, hi)) {
				swap(lo, hi);
				alternateTape(activeTape);
		}	
		dest.setF(lo);
	}
	
}

TwoWayMerge::TwoWayMerge() {}

char TwoWayMerge::doIt(const char *filename, int dataSize)
{
	split(filename, dataSize);
	merge();
	
	
	return 1;
}

void TwoWayMerge::split(const char *filename, int dataSize)
{
	_dataSize = dataSize;
	tapeSrc1.openIn(filename);
	_dataLength = tapeSrc1.fileEnd() / dataSize;
	std::cout<< "input file has "<<_dataLength<<" data\n";
	
	int numBlocks = _dataLength / BLOCKSIZE + 1;
	std::cout << "break into " << numBlocks << " blocks\n";
	
	tapeDst1.openOut("./tapeb1.b");
	tapeDst2.openOut("./tapeb2.b");
	
	int run1 = 0;
	int run2 = 0;

	char *block = new char[BLOCKSIZE * 4];
	for(int i = 0; i < numBlocks; i++) {
		int readSize = tapeSrc1.readBlock(i * BLOCKSIZE * 4, BLOCKSIZE * 4, block);
		
		mySortFunc((float*)block);

		if(i%2 == 0) {
			tapeDst1.writeBlock(readSize, block);
			run1++;
		}
		else {
			tapeDst2.writeBlock(readSize, block);
			run2++;
		}
	}
	
	tapeSrc1.close();
	tapeDst1.close();
	tapeDst2.close();
	delete[] block;
	
	std::cout<< "tape 1 has "<<run1<<" runs\n";
	std::cout<< "tape 2 has "<<run2<<" runs\n";
}

void TwoWayMerge::merge()
{
	std::cout<< "merging...";
	int runLength = BLOCKSIZE;
	
	char b2a = 1;
	while(_dataLength > runLength) {
		
		int numRuns = _dataLength / runLength + 1;
		int runStart = 0;
		
		int realRun = runLength;
		if(realRun > _dataLength)
			realRun = _dataLength;
			
		std::cout << "run length " << realRun << "\n";
		std::cout<< "num runs "<< numRuns<<"\n";
		
		if(b2a) {
			std::cout<< "tape b to a\n";
			b2a = 0;
			tapeSrc1.openIn("./tapeb1.b");
			tapeSrc2.openIn("./tapeb2.b");
			tapeDst1.openOut("./tapea1.b");
			tapeDst2.openOut("./tapea2.b");
		}
		else {
			std::cout<< "tape a to b\n";
			b2a = 1;
			
			tapeSrc1.openIn("./tapea1.b");
			tapeSrc2.openIn("./tapea2.b");
			tapeDst1.openOut("./tapeb1.b");
			tapeDst2.openOut("./tapeb2.b");
		}
		
#if defined DBG	
		tapeSrc1.print(runLength);
		tapeSrc2.print(runLength);
#endif
			
		int numRunOn1 = numRuns / 2;
		numRunOn1 += numRuns - numRunOn1 * 2;
			
		std::cout<<"run on first tape "<<numRunOn1<<"\n";
		
		for(int i = 0; i < numRunOn1 ; i++) {
#if defined DBG
			std::cout<<"run "<<i<<"\n";
#endif
			if(i%2 == 0) {
				mergeTapes(tapeSrc1, tapeSrc2, tapeDst1, realRun, runStart);
				
			}
			else {
				mergeTapes(tapeSrc1, tapeSrc2, tapeDst2, realRun, runStart);
				
			}
			runStart += realRun * 4;
		}
		
		tapeDst1.finalize();
		tapeDst2.finalize();
		
		tapeSrc1.close();
		tapeSrc2.close();
		tapeDst1.close();
		tapeDst2.close();

		int sdst = tapeDst1.readLocation(4) + tapeDst2.readLocation(4);
		if(sdst != _dataLength) {
			std::cout<< "not all floats saved! " << sdst << " : " << _dataLength * 4 << "\n";
			exit(0);
		}
		
#if defined DBG


		tapea1.print(runLength*2);
		tapea2.print(runLength*2);
#endif
		_resultFileName = tapeDst1.fileName();
			
		runLength = runLength * 2;
	}

}

char TwoWayMerge::validate()
{
	tapeSrc1.openIn(_resultFileName.c_str());
	int end = tapeSrc1.fileEnd();
	
	if(end != _dataLength * _dataSize) {
		std::cout<< "not enought data in "<< _resultFileName << "\n";
		std::cout<< "require " << _dataLength * _dataSize << "\n";
		std::cout<< " " << end << " available\n";
		return 0;
	}
	std::cout<<"varidating result...\n";
	tapeSrc1.setReadLimit(0, end);
	float pre, cur;
	tapeSrc1.getF(pre);
	for(int i = 1; i < _dataLength; i++)
	{
		tapeSrc1.getF(cur);
		if(cur < pre) {
			std::cout<<" " <<  tapeSrc1.readLocation(_dataSize)<<" sort error! "<< cur << " < " << pre <<" !\n";
			return 0;
		}
		pre = cur;
	}
	tapeSrc1.close();
	std::cout<<"passed!\n";
	return 1;
}
