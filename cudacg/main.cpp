/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implements a conjugate graident solver on GPU
 * using CUBLAS and CUSPARSE
 * 
 */

// #include <cutil_inline.h>
#include <cusparse.h>
#include <cublas.h>
#include <radixsort_implement.h>
#include <cuReduceSum_implement.h>
#include <CudaReduction.h>
#include <CUDABuffer.h>
#include <BaseBuffer.h>
#include <CudaScan.h>

cudaEvent_t start_event, stop_event;
    
// #define FULLREDUCTION
#define USEREDFN
void testReduceSum()
{
// float sum is not exactly precise, more numbers, larger the sum
// less accurate it gets
//
// n 524285 blocks x threads : 128 x 512 sharedmem size 2048
// sum: 6.28856e+06  proof: 6.28854e+06.
// n 4093 blocks x threads : 4 x 512 sharedmem size 2048
// sum: 4087.44  proof: 4087.44.
	std::cout<<"\n test reduce sum:\n";
    const uint m = (1<<17)-3;
    float * h_data = new float[m];
    
    uint roundedSize = (m * 4);
    
    float * d_idata;
    cudaMalloc((void**)&d_idata, roundedSize);
    
    uint i;
    std::cout<<" generating "<<m<<" random numbers...\n";
    float proof = 0.f;
    for(i=0; i< m; i++) {
        h_data[i] = (((float)rand()/RAND_MAX) + .5f) * 1.0005f;
        proof += h_data[i];
    }
    
    std::cout<<" transfer idata to device\n";
    
    cudaMemcpy(d_idata, h_data, m * 4, cudaMemcpyHostToDevice);
    
    std::cout<<" reduce sum on gpu\n";
#ifdef USEREDFN    
    std::cout<<" use red\n";
    CudaReduction red;
    red.initOnDevice();
    
    float redsum;
    red.sum<float>(redsum, d_idata, m);
    std::cout<<" sum: "<<redsum<<" ";
#else   
    uint threads, blocks;
    uint n = m;
	getReduceBlockThread(blocks, threads, n);
	
	std::cout<<"n "<<n<<" blocks x threads : "<<blocks<<" x "<<threads<<" sharedmem size "<<threads * sizeof(float)<<"\n";
	
	float * d_odata;
    cudaMalloc((void**)&d_odata, blocks * 4);
    
	cuReduce_F_Sum(d_odata, d_idata, m, blocks, threads);
	
#ifdef FULLREDUCTION
    n = blocks;	
	while(n > 1) {
	    blocks = threads = 0;
		getReduceBlockThread(blocks, threads, n);
		
		std::cout<<"n "<<n<<" blocks x threads : "<<blocks<<" x "<<threads<<" sharedmem size "<<threads * sizeof(float)<<"\n";
	
		cuReduce_F_Sum(d_odata, d_odata, n, blocks, threads);
		
		n = (n + (threads*2-1)) / (threads*2);
	}
	float sum;
	cudaMemcpy(&sum, d_odata, 4, cudaMemcpyDeviceToHost);
	
#else
	float h_odata[ReduceMaxBlocks];
	cudaMemcpy(h_odata, d_odata, blocks * 4, cudaMemcpyDeviceToHost);
	float sum = 0.f;
	for(i=0; i< blocks;i++)
	    sum += h_odata[i];
#endif
	std::cout<<" sum: "<<sum<<"\n";
#endif
	std::cout<<" proof: "<<proof<<"\n";
}

void makeRandomUintVector(KeyValuePair *a, unsigned int numElements, unsigned int keybits)
{
    // Fill up with some random data
    int keyshiftmask = 0;
    if (keybits > 16) keyshiftmask = (1 << (keybits - 16)) - 1;
    int keymask = 0xffff;
    if (keybits < 16) keymask = (1 << keybits) - 1;

    srand(95123);
    for(unsigned int i=0; i < numElements; ++i)   
    { 
        a[i].key = ((rand() & keyshiftmask)<<16) | (rand() & keymask); 
		a[i].value = i;
    }
}

void testRadixSort()
{
    printf("test radix sort\n");
    
    unsigned int numElements = 1048576;
    unsigned int numIterations = (numElements >= 16777216) ? 10 : 100;
    
    KeyValuePair *h_data = (KeyValuePair *)malloc(numElements*sizeof(KeyValuePair));
    
    makeRandomUintVector(h_data, numElements, 12);
	
    KeyValuePair *d_data;
    cudaMalloc((void **)&d_data, numElements*sizeof(KeyValuePair));
	
	cudaMemcpy(d_data, h_data, numElements * sizeof(KeyValuePair), cudaMemcpyHostToDevice);
    
	KeyValuePair *d_data1;
    cudaMalloc((void **)&d_data1, numElements*sizeof(KeyValuePair));
	
    RadixSort(d_data, d_data1, numElements, 18);
    
    cudaMemcpy(h_data, d_data, numElements * sizeof(KeyValuePair), cudaMemcpyDeviceToHost);
    
	char passed = 1;
    for(unsigned int i=0; i<numElements-1; ++i)
    {
		// printf("%4d : %4d\n", h_data[i].key, h_data[i].value);
        if( (h_data[i].key)>(h_data[i+1].key) )
        {
            printf("Unordered key[%d]: %d > key[%d]: %d\n", i, h_data[i].key, i+1, h_data[i+1].key);
            passed = 0;
			break;
	    }
    }
	
	if(passed) printf("passed!\n");

    cudaFree(d_data);
    cudaFree(d_data1);
    free(h_data);
}

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, float *val, int M, int N, int nz)
{
    I[0] = 0, J[0] = 0, J[1] = 1;
    val[0] = (float)rand()/RAND_MAX + 10.0f;
    val[1] = (float)rand()/RAND_MAX;
    int start;
    for (int i = 1; i < N; i++) {
        if (i > 1) 
            I[i] = I[i-1]+3;
        else 
            I[1] = 2;
        start = (i-1)*3 + 2;
        J[start] = i - 1;
        J[start+1] = i;
        if (i < N-1) 
            J[start+2] = i + 1;
        val[start] = val[start-1];
        val[start+1] = (float)rand()/RAND_MAX + 10.0f;
        if (i < N-1) 
            val[start+2] = (float)rand()/RAND_MAX;
    }
    I[N] = nz;
}

void testCg()
{
        int M = 0, N = 0, nz = 0, *I = NULL, *J = NULL;
    float *val = NULL;
    const float tol = 1e-5f;   
    const int max_iter = 10000;
    float *x; 
    float *rhs; 
    float a, b, r0, r1;
    int *d_col, *d_row;
    float *d_val, *d_x;
    float *d_r, *d_p, *d_Ax;
    int k;
    
    /* Generate a random tridiagonal symmetric matrix in CSR format */
    M = N = 1048576;
    nz = (N-2)*3 + 4;
    I = (int*)malloc(sizeof(int)*(N+1));
    J = (int*)malloc(sizeof(int)*nz);
    val = (float*)malloc(sizeof(float)*nz);
    printf("generating tridiagonal symmetric matrix\n");
    genTridiag(I, J, val, M, N, nz);

    x = (float*)malloc(sizeof(float)*N);
    rhs = (float*)malloc(sizeof(float)*N);

    for (int i = 0; i < N; i++) {
        rhs[i] = 1.0;
        x[i] = 0.0;
    }
    
    cusparseHandle_t handle = 0;
    cusparseStatus_t status;
    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("!!!! CUSPARSE initialization error\n" );
        exit(1);
    }
    
    cusparseMatDescr_t descr = 0;
    status = cusparseCreateMatDescr(&descr); 
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("!!!! CUSPARSE cusparseCreateMatDescr error\n" );
        exit(1);
    }
    
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    
    printf("sending data to device\n");
    
    cudaMalloc((void**)&d_col, nz*sizeof(int));
    cudaMalloc((void**)&d_row, (N+1)*sizeof(int));
    cudaMalloc((void**)&d_val, nz*sizeof(float));
    cudaMalloc((void**)&d_x, N*sizeof(float));  
    cudaMalloc((void**)&d_r, N*sizeof(float));
    cudaMalloc((void**)&d_p, N*sizeof(float));
    cudaMalloc((void**)&d_Ax, N*sizeof(float));

    cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val, nz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice);

    // cusparseScsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, 1.0, descr, d_val, d_row, d_col, d_x, 0.0, d_Ax);
    cublasSaxpy(N, -1.0, d_Ax, 1, d_r, 1);
    r1 = cublasSdot(N, d_r, 1, d_r, 1);
    
    printf("solving\n");
    k = 1;
    while (r1 > tol*tol && k <= max_iter) {
        if (k > 1) {
            b = r1 / r0;
            
// http://docs.nvidia.com/cuda/cublas/index.html#cublas-level-1-function-reference
//  cublasSscal(cublasHandle_t handle, int n,
//              const float           *alpha,
//              float           *x, int incx)
//  x [ j ] = alpha * x [ j ] for i = 1 , ... , n and j = 1 + ( i - 1 ) *  incx

            cublasSscal(N, b, d_p, 1);
            
// cublasSaxpy(cublasHandle_t handle, int n,
//                           const float           *alpha,
//                         const float           *x, int incx,
//                         float                 *y, int incy)
//  y [ j ] = alpha * x [ k ] + y [ j ] for i = 1 , ... , n , k = 1 + ( i - 1 ) *  incx and j = 1 + ( i - 1 ) *  incy

            cublasSaxpy(N, 1.0, d_r, 1, d_p, 1);
        } else {

// cublasScopy(cublasHandle_t handle, int n,
//                         const float           *x, int incx,
//                         float                 *y, int incy)
//  y [ j ] = x [ k ] for i = 1 , ... , n , k = 1 + ( i - 1 ) *  incx and j = 1 + ( i - 1 ) *  incy

            cublasScopy(N, d_r, 1, d_p, 1);
        }
        
// http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrmv
// cusparseScsrmv(cusparseHandle_t handle, cusparseOperation_t transA, 
               // int m, int n, int nnz, const float           *alpha, 
               // const cusparseMatDescr_t descrA, 
               // const float           *csrValA, 
               // const int *csrRowPtrA, const int *csrColIndA,
               // const float           *x, const float           *beta, 
               // float           *y)
//  y = alpha * op ( A ) * x + beta * y

        // cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, 1.0, descr, d_val, d_row, d_col, d_p, 0.0, d_Ax);
        a = r1 / cublasSdot(N, d_p, 1, d_Ax, 1);
        cublasSaxpy(N, a, d_p, 1, d_x, 1);
        cublasSaxpy(N, -a, d_Ax, 1, d_r, 1);

        r0 = r1;
        r1 = cublasSdot(N, d_r, 1, d_r, 1);
        cudaThreadSynchronize();
        printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }
    
    cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);

    float rsum, diff, err = 0.0;
    for (int i = 0; i < N; i++) {
        rsum = 0.0;
        for (int j = I[i]; j < I[i+1]; j++) {
            rsum += val[j]*x[J[j]];
        }
        diff = fabs(rsum - rhs[i]);
        if (diff > err) err = diff;
    }
    
    printf("Test Summary: Errors = %f\n", err);
    printf("%s\n", (k <= max_iter) ? "PASSED" : "FAILED");
    cusparseDestroy(handle);

    free(I);
    free(J);
    free(val);
    free(x);
    free(rhs);
    cudaFree(d_col);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ax);

    cudaThreadExit();
}

void testBuffer()
{
	std::cout<<" test cu buffer\n";
	CUDABuffer db;
	db.create(1024*1024*4);
	
	BaseBuffer hb;
	hb.create(1024*1024*4);
	
	unsigned * h = (unsigned *)hb.data();
	unsigned i;
	for(i=0; i< 1024*1024; i++) h[i] = i+1;
	
	db.hostToDevice(hb.data());
	
	db.deviceToHost(hb.data());
	
	std::cout<<" h[17] "<<h[17];
}

void testReduceMax()
{
	std::cout<<" test find max\n";
	CUDABuffer db;
	db.create(1024*1024*4);
	
	BaseBuffer hb;
	hb.create(1024*1024*4);
	
	unsigned * h = (unsigned *)hb.data();
	unsigned i;
	for(i=0; i< 1024*1024; i++) h[i] = i+1;
	
	db.hostToDevice(hb.data());
	
	CudaReduction reducer;
	reducer.initOnDevice();
	
	cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync);
    cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync);
	cudaEventRecord(start_event, 0);
	int res = 0;
	reducer.max<int>(res, (int *)db.bufferOnDevice(), 1024 * 1024);
	cudaEventRecord(stop_event, 0);
	cudaEventSynchronize(stop_event);
	float met;
	cudaEventElapsedTime(&met, start_event, stop_event);
	std::cout<<" reduction took "<<met<<" milliseconds\n";
	std::cout<<" max "<<res<<"\n";
	cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
}

void testReduceMin()
{
	std::cout<<" test find min\n";
	CUDABuffer db;
	db.create(1024*1024*4);
	
	BaseBuffer hb;
	hb.create(1024*1024*4);
	
	unsigned * h = (unsigned *)hb.data();
	unsigned i;
	for(i=0; i< 1024*1024; i++) h[i] = i+1;
	
	db.hostToDevice(hb.data());
	
	CudaReduction reducer;
	reducer.initOnDevice();
	
	cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync);
    cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync);
	cudaEventRecord(start_event, 0);
	int res = 0;
	reducer.min<int>(res, (int *)db.bufferOnDevice(), 1024 * 1024);
	cudaEventRecord(stop_event, 0);
	cudaEventSynchronize(stop_event);
	float met;
	cudaEventElapsedTime(&met, start_event, stop_event);
	std::cout<<" reduction took "<<met<<" milliseconds\n";
	std::cout<<" min "<<res<<"\n";
	cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
}

void testReduceMinMax()
{
	std::cout<<" test find min max f\n";
	unsigned m = 1024*1024;
	CUDABuffer db;
	db.create(m*4);
	
	BaseBuffer hb;
	hb.create(m*4);
	
	float * h = (float *)hb.data();
	unsigned i;
	for(i=0; i< m; i++) h[i] = -599.f + 999.f * ((float)(rand() & 255))/127.f;
	
	db.hostToDevice(hb.data());
	
	CudaReduction reducer;
	reducer.initOnDevice();
	
	cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync);
    cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync);
	cudaEventRecord(start_event, 0);
	float res[2];
	reducer.minMax<float2, float>(res, (float *)db.bufferOnDevice(), m);
	
	std::cout<<" minmax f "<<res[0]<<","<<res[1]<<"\n";
	
	cudaEventRecord(stop_event, 0);
	cudaEventSynchronize(stop_event);
	float met;
	cudaEventElapsedTime(&met, start_event, stop_event);
	std::cout<<" reduction took "<<met<<" milliseconds\n";
	
	
	cudaEventRecord(start_event, 0);
	
	float smn, smx;
	reducer.min<float>(smn, (float *)db.bufferOnDevice(), m);
	reducer.max<float>(smx, (float *)db.bufferOnDevice(), m);
	
	std::cout<<" min max f "<<smn<<","<<smx<<"\n";
	
	cudaEventRecord(stop_event, 0);
	cudaEventSynchronize(stop_event);
	
	cudaEventElapsedTime(&met, start_event, stop_event);
	std::cout<<" reduction took "<<met<<" milliseconds\n";
	
	cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
}

void testReduceMinMaxBox()
{
	std::cout<<" test find min max f\n";
	unsigned m = 1024*1024;
	CUDABuffer db;
	db.create(m*32);
	
	BaseBuffer hb;
	hb.create(m*32);
	
	Aabb * h = (Aabb *)hb.data();
	float x, y, z;
	unsigned i;
	for(i=0; i< m; i++) {
	    x = 440.f * ((float)(rand() & 255))/256.f - 22.f;
	    y = 44.f * ((float)(rand() & 255))/256.f - 220.f;
	    z = 440.f * ((float)(rand() & 255))/256.f - 2200.f;
	    h[i].low.x = x - 1.f;
	    h[i].low.y = y - 1.f;
	    h[i].low.z = z - 1.f;
	    h[i].high.x = x + 1.f;
	    h[i].high.y = y + 1.f;
	    h[i].high.z = z + 1.f;
	}
	
	db.hostToDevice(hb.data());
	
	CudaReduction reducer;
	reducer.initOnDevice();
	
	cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync);
    cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync);
    
    float res[6];

    for(i=10; i<= 20; i++) {
        cudaEventRecord(start_event, 0);
        
        reducer.minMaxBox<Aabb, float3>((Aabb *)&res, (float3 *)db.bufferOnDevice(), 2<<i);
        
        std::cout<<" min max box ("<<res[0]<<","<<res[1]<<","<<res[2]<<"),("
                <<res[3]<<","<<res[4]<<","<<res[5]<<")\n";
        
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        float met;
        cudaEventElapsedTime(&met, start_event, stop_event);
        std::cout<<" reduction "<<(1<<i)<<" boxes took "<<met<<" milliseconds\n";
	}
	cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
}

void testScan()
{
	std::cout<<" test scan\n";
	unsigned m = 1024*1024;
		
	CUDABuffer dcount;
	dcount.create(m*4);
	
	CUDABuffer dsum;
	dsum.create(m*4);
	
	BaseBuffer hb;
	hb.create(m*4);
	
	unsigned * h = (unsigned *)hb.data();
	unsigned i;
	for(i=0; i< m; i++) {
	    h[i] = !(rand() & 3);
	}
	
	dcount.hostToDevice(hb.data());
	
	CudaScan csn;
	csn.create(m);
	
	cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync);
    cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync);
	
	for(i=10; i< 20; i++) {
		cudaEventRecord(start_event, 0);
	
		unsigned result = csn.prefixSum(&dsum, &dcount, 2<<i);
		
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		float met;
		cudaEventElapsedTime(&met, start_event, stop_event);
		std::cout<<" scan "<<(2<<i)<<" ints took "<<met<<" milliseconds\n";
		std::cout<<" result is "<<result<<"\n";
	}
	cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
}

int main(int argc, char **argv)
{
    // This will pick the best possible CUDA capable device
    cudaDeviceProp deviceProp;
    int devID = 0;
    
    cudaGetDeviceProperties(&deviceProp, devID);

    // Statistics about the GPU device
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n", 
		deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    int version = (deviceProp.major * 0x10 + deviceProp.minor);
    if(version < 0x11) 
    {
        printf("cudacg: requires a minimum CUDA compute 1.1 capability\n");
        printf("PASSED");
        cudaThreadExit();
        exit(1);
    }
        
    // printf("test conjugate gradient\n");
    // testCg();
    testReduceMin();
	testReduceMinMax();
	testReduceMinMaxBox();
	testReduceSum();
	testScan();
    testRadixSort();
    printf("done.\n");
    exit(0);
}
