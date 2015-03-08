#ifndef SCANUTIL_H
#define SCANUTIL_H
class CUDABuffer;
class ScanUtil {
public:
	ScanUtil();
	static unsigned getScanResult(CUDABuffer * counts, 
	                                CUDABuffer * sums, 
	                                unsigned bufferLength);
private:
};
#endif        //  #ifndef SCANUTIL_H

