#ifndef CUDAMASSSYSTEM_H
#define CUDAMASSSYSTEM_H

/*
 *  CudaMassSystem.h
 *  testcudafem
 *
 *  Created by jian zhang on 6/9/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <MassSystem.h>
class CUDABuffer;

class CudaMassSystem : public MassSystem {
public:
	CudaMassSystem();
	virtual ~CudaMassSystem();
	
	virtual void initOnDevice();
	
	void setDeviceXPtr(CUDABuffer * ptr, unsigned loc);
	void setDeviceXiPtr(CUDABuffer * ptr, unsigned loc);
	void setDeviceVPtr(CUDABuffer * ptr, unsigned loc);
	void setDeviceMassPtr(CUDABuffer * ptr, unsigned loc);
	void setDeviceTretradhedronIndicesPtr(CUDABuffer * ptr, unsigned loc);

    void * deviceX();
	void * deviceXi();
	void * deviceV();
	void * deviceMass();
	void * deviceAnchor();
	void * deviceTretradhedronIndices();
	
	void sendXToHost();
	void sendVToHost();
protected:
	CUDABuffer * deviceXBuf();
	CUDABuffer * deviceVBuf();
	const unsigned xLoc() const;
	const unsigned vLoc() const;
	CUDABuffer * anchorBuf();

private:
	CUDABuffer * m_deviceX;
	CUDABuffer * m_deviceXi;
	CUDABuffer * m_deviceV;
	CUDABuffer * m_deviceMass;
	CUDABuffer * m_deviceAnchor;
	CUDABuffer * m_deviceTetrahedronIndices;
	unsigned m_xLoc, m_xiLoc, m_vLoc, m_massLoc, m_iLoc;
};
#endif        //  #ifndef CUDAMASSSYSTEM_H