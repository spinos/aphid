#include "QuickSort.h"

void QuickSort::Sort(BaseArray &array,int first,int last)
{
	if(last < first) return;
	
	int low,high;
	float list_separator;

	low = first;
	high = last;
	list_separator = array.sortKeyAt((first+last)/2);
	do
	{
		while(array.sortKeyAt(low)<list_separator) low++;
		while(array.sortKeyAt(high)>list_separator) high--;

		if(low<=high)
		{
			/*temp = array[low];
			array[low++] = array[high];
			array[high--]=temp;*/
			array.swapElement(low, high);
			low++;
			high--;
		}
	} while(low<=high);
	
	if(first<high) Sort(array,first,high);
	if(low<last) Sort(array,low,last);
}

void QuickSort::Sort(vector<unsigned> &array,int first,int last)
{
	if(last < first) return;
	
	int low,high;
	float list_separator;
	unsigned temp;

	low = first;
	high = last;
	list_separator = array[(first+last)/2];
	do
	{
		while(array[low] < list_separator) low++;
		while(array[high] > list_separator) high--;

		if(low<=high)
		{
			temp = array[low];
			array[low++] = array[high];
			array[high--]=temp;
		}
	} while(low<=high);
	
	if(first<high) Sort(array,first,high);
	if(low<last) Sort(array,low,last);
}
//:~