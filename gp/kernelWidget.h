/*
 *  kernelWidget.h
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef GPR_KERNEL_WIDGET_H
#define GPR_KERNEL_WIDGET_H

#include <Plot2DWidget.h>

/// forward declear cannot include linear math header here
namespace lfr {
template<typename T>
class DenseMatrix;
}

namespace aphid {
namespace gpr {
    
class KernelWidget : public Plot2DWidget {

	Q_OBJECT
	
public:
	KernelWidget(QWidget *parent = 0);
	virtual ~KernelWidget();
	
	void plotK(const lfr::DenseMatrix<float> * K);
	
protected:

private:

};

}
}
#endif
