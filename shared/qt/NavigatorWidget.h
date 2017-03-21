/*
 *   NavigatorWidget.h
 *
 *   image preview
 */
 
#ifndef APH_NAVIGATOR_WIDGET_H
#define APH_NAVIGATOR_WIDGET_H

#include <qt/Plot2DWidget.h>

namespace aphid {

class ExrImage;

class NavigatorWidget : public aphid::Plot2DWidget {

	Q_OBJECT
	
public:
	NavigatorWidget(QWidget *parent = 0);
	NavigatorWidget(const ExrImage * img,
				QWidget *parent = 0);
	virtual ~NavigatorWidget();
	
	void setImage(const Array3<float> & img);
	
protected:
	virtual void processCamera(QMouseEvent *event);
	
private:

};

}
#endif
