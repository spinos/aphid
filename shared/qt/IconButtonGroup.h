/*
 *  IconButtonGroup.h
 *  
 *  push button with label, icon, name_id
 *
 *  Created by jian zhang on 9/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_ICON_BUTTON_GROUP_H
#define APH_ICON_BUTTON_GROUP_H

#include <QPushButton>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

namespace aphid {

class IconButtonGroup : public QPushButton 
{
	Q_OBJECT
public:
	IconButtonGroup(const QIcon& icon,
		const QString & name, QWidget *parent = 0);
	
	void setNameId(int x);
	const int& nameId() const;
	
private slots:
	void sendPressedValue();
	
signals:
	void buttonPressed2(QPair<int, int> x);
	
private slots:
	
private:
	int m_nameId;
	
};

}
#endif