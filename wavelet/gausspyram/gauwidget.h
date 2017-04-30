/*
 *   gauwidget.h
 *
 */
 
#ifndef GAU_WIDGET_H
#define GAU_WIDGET_H

#include <qt/Plot2DWidget.h>

namespace aphid {

class ExrImage;

template<typename T>
class Array3;

class UniformPlot2DImage;

namespace img {

template<typename T>
class GaussianPyramid;

class HeightField;

}

}

class GauWidget : public aphid::Plot2DWidget {

	Q_OBJECT
	
public:
	GauWidget(const aphid::ExrImage * img,
				QWidget *parent = 0);
	virtual ~GauWidget();
	
	void resizeEvent(QResizeEvent *event);
	
protected:

public slots:

private:
	void resample();
	
private:
	aphid::UniformPlot2DImage * m_plotY;
	aphid::Array3<float> * m_Y;
	aphid::img::HeightField * m_gau;
	
};

#endif
