/*
 *  DictionaryView.h
 *  
 *
 *  Created by jian zhang on 9/9/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef LFR_DICTIONARY_VIEW_H
#define LFR_DICTIONARY_VIEW_H

#include <BaseImageWidget.h>

namespace lfr {

class DictionaryView : public aphid::BaseImageWidget {

	Q_OBJECT
	
public:
	DictionaryView(QWidget *parent = 0);
	virtual ~DictionaryView();

public slots:
   void recvDictionary(const QImage &image);
   	
protected:
	virtual void clientDraw(QPainter * pr);

private:
	QPixmap m_pixmap;
	
};

}
#endif
