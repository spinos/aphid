/*
 *   dthfwidget.h
 *
 */
 
#ifndef DTHF_WIDGET_H
#define DTHF_WIDGET_H

#include <qt/Plot2DWidget.h>

namespace aphid {

class ExrImage;

template<typename T>
class Array3;

class UniformPlot2DImage;

namespace wla {

class DualTree2;

}

}

class DthfWidget : public aphid::Plot2DWidget {

	Q_OBJECT
	
public:
	DthfWidget(const aphid::ExrImage * img,
				QWidget *parent = 0);
	virtual ~DthfWidget();
	
	void resizeEvent(QResizeEvent *event);
	
protected:

public slots:
	void recvL0scale(double x);
	void recvL1scale(double x);
	void recvL2scale(double x);
	void recvL3scale(double x);

private:
	void checkSynthesisErr(const aphid::Array3<float> & synth);
	void resynthsize();
	
private:
/// input signal
	aphid::Array3<float> * m_inputX;
	aphid::wla::DualTree2 * m_synthesis;
	aphid::UniformPlot2DImage * m_plot;
	float m_levelScale[4];
	
};

#endif
