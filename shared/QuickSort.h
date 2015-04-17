#ifndef QUICKSORT_H
#define QUICKSORT_H
#include <iostream>
#include <stdlib.h>
#include <BaseArray.h>
#include <deque>
using namespace std;

template <typename KeyType, typename ValueType>
struct QuickSortPair {
   KeyType key;
   ValueType value;
};

template <typename KeyType, typename ValueType>
class QuickSort1
{
public:
    static void Sort(QuickSortPair<KeyType, ValueType > * kv, int first, int last)
    {
        if(last < first) return;
        int low, high;
        QuickSortPair<KeyType, ValueType > temp;
        low = first;
        high = last;
        KeyType list_separator = kv[(first+last)/2].key;
        do
        {
            while(kv[low].key < list_separator) low++;
            while(kv[high].key > list_separator) high--;
    
            if(low<=high)
            {
                temp = kv[low];
                kv[low++] = kv[high];
                kv[high--]=temp;
            }
        } while(low<=high);
        
        if(first<high) Sort(kv,first,high);
        if(low<last) Sort(kv,low,last);
    }
};

class QuickSort
{
public:
	QuickSort() {}
	~QuickSort() {}
	
	static void Sort(unsigned * kv, int first, int last);
	static void Sort(BaseArray &array,int first,int last);
	static void Sort(vector<unsigned> &array,int first,int last);
	static void Sort(deque<unsigned> &array,int first,int last);
};
#endif        //  #ifndef QUICKSORT_H
