#ifndef CU_TESTATOMIC_IMPL_H
#define CU_TESTATOMIC_IMPL_H

extern "C" {
void cu_testAtomic(int * obin,
                    int * idata,
                    int h,
                    int n);

void cu_setZero(int * obin,
                    int n);

void cu_addOne(int * obin,
                    int n);

}

#endif        //  #ifndef CU_TESTATOMIC_IMPL_H

