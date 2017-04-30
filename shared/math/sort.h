/*
 *  sort.h
 *  
 *
 *  Created by jian zhang on 12/19/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_MATH_SORT_H
#define APH_MATH_SORT_H
namespace aphid {

template <typename T, typename I>
void sort(I* irOut, T* prOut,I beg, I end) {
   I i;
   if (end <= beg) return;
   I pivot=beg;
   for (i = beg+1; i<=end; ++i) {
      if (irOut[i] < irOut[pivot]) {
         if (i == pivot+1) {
            I tmp = irOut[i];
            T tmpd = prOut[i];
            irOut[i]=irOut[pivot];
            prOut[i]=prOut[pivot];
            irOut[pivot]=tmp;
            prOut[pivot]=tmpd;
         } else {
            I tmp = irOut[pivot+1];
            T tmpd = prOut[pivot+1];
            irOut[pivot+1]=irOut[pivot];
            prOut[pivot+1]=prOut[pivot];
            irOut[pivot]=irOut[i];
            prOut[pivot]=prOut[i];
            irOut[i]=tmp;
            prOut[i]=tmpd;
         }
         ++pivot;
      }
   }
   sort(irOut,prOut,beg,pivot-1);
   sort(irOut,prOut,pivot+1,end);
}

template <typename T, typename I>
void sort_descent(I* irOut, T* prOut,I beg, I end) {
   I i;
   if (end <= beg) return;
   I pivot=beg;
   for (i = beg+1; i<=end; ++i) {
      if (irOut[i] > irOut[pivot]) {
         if (i == pivot+1) {
            I tmp = irOut[i];
            T tmpd = prOut[i];
            irOut[i]=irOut[pivot];
            prOut[i]=prOut[pivot];
            irOut[pivot]=tmp;
            prOut[pivot]=tmpd;
         } else {
            I tmp = irOut[pivot+1];
            T tmpd = prOut[pivot+1];
            irOut[pivot+1]=irOut[pivot];
            prOut[pivot+1]=prOut[pivot];
            irOut[pivot]=irOut[i];
            prOut[pivot]=prOut[i];
            irOut[i]=tmp;
            prOut[i]=tmpd;
         }
         ++pivot;
      }
   }
   sort_descent(irOut,prOut,beg,pivot-1);
   sort_descent(irOut,prOut,pivot+1,end);
}

}
#endif