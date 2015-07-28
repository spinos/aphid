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
    void setDeviceVaPtr(CUDABuffer * ptr, unsigned loc);
	void setDeviceMassPtr(CUDABuffer * ptr, unsigned loc);
	void setDeviceAnchorPtr(CUDABuffer * ptr, unsigned loc);
	void setDeviceTretradhedronIndicesPtr(CUDABuffer * ptr, unsigned loc);

    void * deviceX();
	void * deviceXi();
	void * deviceV();
    void * deviceVa();
	void * deviceMass();
	void * deviceAnchor();
	void * deviceTretradhedronIndices();
	void * deviceInitialMass();
	
	void sendXToHost();
	void sendVToHost();
	
protected:
	CUDABuffer * deviceXBuf();
	CUDABuffer * deviceVBuf();
    CUDABuffer * deviceAnchoredVBuf();
    CUDABuffer * deviceAnchorBuf();
	const unsigned xLoc() const;
	const unsigned vLoc() const;
    const unsigned anchoredVLoc() const;
	CUDABuffer * anchorBuf();
	virtual void updateMass();
private:
	CUDABuffer * m_deviceX;
	CUDABuffer * m_deviceXi;
	CUDABuffer * m_deviceV;
    CUDABuffer * m_deviceVa;
	CUDABuffer * m_deviceMass;
	CUDABuffer * m_deviceAnchor;
	CUDABuffer * m_deviceTetrahedronIndices;
	CUDABuffer * m_initialMass;
	unsigned m_xLoc, m_xiLoc, m_vLoc, m_vaLoc, m_massLoc, m_anchorLoc, m_iLoc;
};
#endif        //  #ifndef CUDAMASSSYSTEM_H
