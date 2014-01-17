/****************************************************************************
**
** Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).
** All rights reserved.
** Contact: Nokia Corporation (qt-info@nokia.com)
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** You may use this file under the terms of the BSD license as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of Nokia Corporation and its Subsidiary(-ies) nor
**     the names of its contributors may be used to endorse or promote
**     products derived from this software without specific prior written
**     permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
** $QT_END_LICENSE$
**
****************************************************************************/

/*
    delegate.cpp

    A delegate that allows the user to change integer values from the model
    using a spin box widget.
*/

#include <QtGui>

#include "delegate.h"
#include "treeitem.h"
#include "ColorEdit.h"

//! [0]
SpinBoxDelegate::SpinBoxDelegate(QObject *parent)
    : QItemDelegate(parent)
{
}
//! [0]

//! [1]
QWidget *SpinBoxDelegate::createEditor(QWidget *parent,
    const QStyleOptionViewItem &/* option */,
    const QModelIndex & index) const
{
	TreeItem *item = static_cast<TreeItem*>(index.internalPointer());
	qDebug()<<"deleg name"<<item->name().c_str()<<" type "<<item->valueType();
    QWidget *editor = 0;
	
	switch (item->valueType()) {
		case 0:
			editor = createIntEditor(parent);
			break;
		case 1:
			editor = createDoubleEditor(parent);
			break;
		case 2:
			editor = createBoolEditor(parent);
			break;
		default:
			editor = createColorEditor(parent);
			break;
	}
    return editor;
}
//! [1]

//! [2]
void SpinBoxDelegate::setEditorData(QWidget *editor,
                                    const QModelIndex &index) const
{
	TreeItem *item = static_cast<TreeItem*>(index.internalPointer());
	QVariant value = index.model()->data(index, Qt::EditRole);
    
	switch (item->valueType()) {
		case 0:
			setIntEditorValue(editor, value); 
			break;
		case 1:
			setDoubleEditorValue(editor, value); 
			break;
		case 2:
			setBoolEditorValue(editor, value); 
			break;
		default:
			setColorEditorValue(editor, value); 
			break;
	}
}
//! [2]

//! [3]
void SpinBoxDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
                                   const QModelIndex &index) const
{
    TreeItem *item = static_cast<TreeItem*>(index.internalPointer());
	switch (item->valueType()) {
		case 0:
			model->setData(index, getIntEditorValue(editor), Qt::EditRole);
			break;
		case 1:
			model->setData(index, getDoubleEditorValue(editor), Qt::EditRole);
			break;
		case 2:
			model->setData(index, getBoolEditorValue(editor), Qt::EditRole);
			break;
		default:
			model->setData(index, getColorEditorValue(editor), Qt::EditRole);
			break;
	}
}
//! [3]

//! [4]
void SpinBoxDelegate::updateEditorGeometry(QWidget *editor,
    const QStyleOptionViewItem &option, const QModelIndex &/* index */) const
{
    editor->setGeometry(option.rect);
}

QString SpinBoxDelegate::translateBoolToStr(const QVariant & src) const
{
	if(src.toBool()) return tr("true");
	return tr("false");
}

QVariant SpinBoxDelegate::translateStrToBool(const QString & src) const
{
	if(src == "on" || src == "1" || src == "true") return QVariant(true);
	return QVariant(false);
}

QWidget * SpinBoxDelegate::createIntEditor(QWidget *parent) const
{
	QLineEdit *editor = new QLineEdit(parent);
	editor->setValidator(&m_intValidate);
    return editor;
}

QWidget * SpinBoxDelegate::createDoubleEditor(QWidget *parent) const
{
	QLineEdit *editor = new QLineEdit(parent);
	editor->setValidator(&m_dlbValidate);
    return editor;
}

QWidget * SpinBoxDelegate::createBoolEditor(QWidget *parent) const
{
    return new QLineEdit(parent);
}

QWidget * SpinBoxDelegate::createColorEditor(QWidget *parent) const
{
	return new ColorEdit(parent);
}

void SpinBoxDelegate::setIntEditorValue(QWidget *editor, QVariant & value) const
{
	QString t;
	int intValue = value.toInt();
	t.setNum(intValue);
	QLineEdit *le = static_cast<QLineEdit*>(editor);
	le->setText(t);
}

void SpinBoxDelegate::setDoubleEditorValue(QWidget *editor, QVariant & value) const
{
	QString t;
	double dlbValue = value.toDouble();
	t.setNum(dlbValue);
	QLineEdit *le = static_cast<QLineEdit*>(editor);
	le->setText(t);
}

void SpinBoxDelegate::setBoolEditorValue(QWidget *editor, QVariant & value) const
{
	QString t = translateBoolToStr(value);
	QLineEdit *le = static_cast<QLineEdit*>(editor);
	le->setText(t);
}

void SpinBoxDelegate::setColorEditorValue(QWidget *editor, QVariant & value) const
{
	ColorEdit *le = static_cast<ColorEdit*>(editor);
	qDebug()<<"in col"<<value.value<QColor>();
	le->setColor(value.value<QColor>());
}

QVariant SpinBoxDelegate::getIntEditorValue(QWidget *editor) const
{
	QLineEdit *le = static_cast<QLineEdit*>(editor);
	int value = le->text().toInt();
	return QVariant(value);
}

QVariant SpinBoxDelegate::getDoubleEditorValue(QWidget *editor) const
{
	QLineEdit *le = static_cast<QLineEdit*>(editor);
	double value = le->text().toDouble();
	return QVariant(value);
}

QVariant SpinBoxDelegate::getBoolEditorValue(QWidget *editor) const
{
	QLineEdit *le = static_cast<QLineEdit*>(editor);
	return QVariant(translateStrToBool(le->text()));
}

QVariant SpinBoxDelegate::getColorEditorValue(QWidget *editor) const
{
	ColorEdit *le = static_cast<ColorEdit*>(editor);
	qDebug()<<"col"<<le->color();
	return QVariant(le->color());
}
