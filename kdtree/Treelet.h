/*
 *  Treelet.h
 *  testntree
 *
 *  Created by jian zhang on 10/31/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

template<int NumLevels>
class Treelet {
	///			[0      1]parent		  level 0
	///------------------------------------------
	///   [2    3]0      [4      5]1      level 1
	///
    ///  [6 7]2 [8 9]3  [10 11]4 [12 13]5 level 2
    ///
	///  []6[]7 []8[]9  []10[]11 []12[]13 level 3
	///------------------------------------------
	
	static int LevelOffset[NumLevels+1];
public:
	Treelet();
	virtual ~Treelet();
	
	int numNodes() const;
	
	static int LastLevelOffset();
	static int OffsetByLevel(int level);
	static int ChildOffset(int x);
protected:
	
};

template<int NumLevels>
int Treelet<NumLevels>::LevelOffset[NumLevels+1];

template<int NumLevels>
Treelet<NumLevels>::Treelet()
{
	int i;
	int a = 0;
	for(i=1;i<=NumLevels;i++) {
		LevelOffset[i] = a;
		a += 1<<i;
	}
}

template<int NumLevels>
Treelet<NumLevels>::~Treelet() {}

template<int NumLevels>
int Treelet<NumLevels>::LastLevelOffset()
{ return LevelOffset[NumLevels]; }

template<int NumLevels>
int Treelet<NumLevels>::OffsetByLevel(int level)
{ return LevelOffset[level]; }

template<int NumLevels>
int Treelet<NumLevels>::numNodes() const
{ return (1<<(NumLevels+1)) - 2; }

template<int NumLevels>
int Treelet<NumLevels>::ChildOffset(int x)
{ return x + 2; }

typedef Treelet<4> Treelet4;
