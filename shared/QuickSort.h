#pragma once

#include <iostream>
#include <stdlib.h>
#include <BaseArray.h>
#include <deque>
using namespace std;

class QuickSort
{
public:
	QuickSort() {}
	~QuickSort() {}
	
	static void Sort(BaseArray &array,int first,int last);
	static void Sort(vector<unsigned> &array,int first,int last);
	static void Sort(deque<unsigned> &array,int first,int last);
};