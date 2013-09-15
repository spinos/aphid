#pragma once

#include <iostream>
#include <stdlib.h>
#include <BaseArray.h>
using namespace std;

class QuickSort
{
public:
	QuickSort() {}
	~QuickSort() {}
	
	static void Sort(BaseArray &array,int first,int last);
	
};