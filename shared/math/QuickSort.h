#ifndef APHID_QUICKSORT_H
#define APHID_QUICKSORT_H

#include <stdlib.h>
#include <vector>
#include <deque>

namespace aphid {

template <typename KeyType, typename ValueType>
struct QuickSortPair {
   KeyType key;
   ValueType value;
};

class QuickSort1
{
public:

	template <typename KeyType, typename ValueType>
	static void SortByKey(ValueType* kv, int first, int last)
	{
		if(last < first) return;
        int low, high;
        ValueType temp;
        low = first;
        high = last;
        KeyType list_separator = kv[(first+last)/2]._key;
        do
        {
            while(kv[low]._key < list_separator) low++;
            while(kv[high]._key > list_separator) high--;
    
            if(low<=high)
            {
                temp = kv[low];
                kv[low++] = kv[high];
                kv[high--]=temp;
            }
        } while(low<=high);
        
        if(first<high) SortByKey<KeyType, ValueType >(kv,first,high);
        if(low<last) SortByKey<KeyType, ValueType >(kv,low,last);
	}
	
	template <typename KeyType, typename ValueType>
	static void SortVector(std::vector<ValueType>& kv, int first, int last)
	{
		if(last < first) return;
        int low, high;
        ValueType temp;
        low = first;
        high = last;
        KeyType list_separator = kv[(first+last)/2]._key;
        do
        {
            while(kv[low]._key < list_separator) low++;
            while(kv[high]._key > list_separator) high--;
    
            if(low<=high)
            {
                temp = kv[low];
                kv[low++] = kv[high];
                kv[high--]=temp;
            }
        } while(low<=high);
        
        if(first<high) SortVector<KeyType, ValueType >(kv,first,high);
        if(low<last) SortVector<KeyType, ValueType >(kv,low,last);
	}
	
    template <typename ValueType>
    static void Sort(ValueType * kv, int first, int last)
    {
        if(last < first) return;
        int low, high;
        ValueType temp;
        low = first;
        high = last;
        ValueType list_separator = kv[(first+last)/2];
        do
        {
            while(kv[low] < list_separator) low++;
            while(kv[high] > list_separator) high--;
    
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
    
    template <typename KeyType, typename ValueType>
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
	static void Sort(std::vector<unsigned> &array,int first,int last);
	static void Sort(std::deque<unsigned> &array,int first,int last);
};

}
#endif        //  #ifndef QUICKSORT_H
