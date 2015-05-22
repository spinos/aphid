#include <cuda_runtime_api.h>
#include <radixsort_implement.h>
#include <iostream>
#include <CudaBase.h>
#include <BaseBuffer.h>
#include <CUDABuffer.h>
#include <CudaDbgLog.h>

CudaDbgLog qslog("1bsort.txt");
cudaEvent_t start_event, stop_event;

void makeRandomUintVector(KeyValuePair *a, unsigned int numElements, unsigned int keybits)
{
    std::cout<<" generating "<<numElements<<" random uint\n";
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

bool checkSortResult(KeyValuePair * a, 
                    unsigned n)
{
    KeyValuePair b = a[0];
    for(unsigned i=1; i<n;i++) {
        if(a[i].key<b.key) {
            std::cout<<" unsorted element["<<i<<"] "<<a[i].key<<" < "<<b.key<<" !\n";
            return false;
        }
        b = a[i];
    }
    return true;
}

void OneBitSort(KeyValuePair *pData0, KeyValuePair *pData1, 
    uint elements,
    uint * counts);

int main(int argc, char **argv)
{
    CudaBase::SetDevice();
    if(!CudaBase::HasDevice) return 1;
    
    unsigned n = (1<<12) - 171;
    BaseBuffer hdata;
    hdata.create(n*8);
    
    KeyValuePair * hostData = (KeyValuePair *)hdata.data();
    makeRandomUintVector(hostData, n, 1);
    
    qslog.writeHash(&hdata, n, "input_unsorted", CudaDbgLog::FOnce);
    
    CUDABuffer ddata;
    ddata.create(n*8);
    ddata.hostToDevice(hostData);
    
    CUDABuffer ddata1;
    ddata1.create(n*8);
    ddata1.hostToDevice(hostData);
    
    CUDABuffer dcounts;
    dcounts.create(8);
    
    cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync);
    cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync);
	cudaEventRecord(start_event, 0);
    
	std::cout<<" launch cu kernel\n";
    OneBitSort((KeyValuePair *)ddata.bufferOnDevice(),
                (KeyValuePair *)ddata1.bufferOnDevice(),
                n,
                (uint *)dcounts.bufferOnDevice());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    float met;
	cudaEventElapsedTime(&met, start_event, stop_event);
	std::cout<<" sort "<<n<<" ints took "<<met<<" milliseconds\n";
		
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    
    qslog.writeHash(&ddata1, n, "output_sorted", CudaDbgLog::FOnce);
    
    ddata1.deviceToHost(hostData);
    if(checkSortResult(hostData, n)) std::cout<<" result form gpu sort passed.\n";
    
	return 0;
}


