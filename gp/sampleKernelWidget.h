/*
 *  sampleKernelWidget.h
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_GPR_SAMPLE_KERNEL_WIDGET_H
#define APH_GPR_SAMPLE_KERNEL_WIDGET_H

#include <Plot1DWidget.h>

namespace aphid {

template<typename T>
class DenseMatrix;

namespace gpr {

class SampleKernelWidget : public Plot1DWidget {

	Q_OBJECT
	
public:
	SampleKernelWidget(const DenseMatrix<float> & yActual, QWidget *parent = 0);
	virtual ~SampleKernelWidget();
	
protected:

private:

};

}
}
#endif
