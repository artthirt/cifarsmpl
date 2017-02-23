#ifndef DRAWCNVWEIGHT_H
#define DRAWCNVWEIGHT_H

#include <QWidget>
#include <QTimer>

#include "custom_types.h"

namespace Ui {
class DrawCnvWeight;
}

class DrawCnvWeight : public QWidget
{
	Q_OBJECT

public:
	explicit DrawCnvWeight(QWidget *parent = 0);
	~DrawCnvWeight();

	void set_weightR(const QVector<QVector<ct::Matf> > &W);
	void set_weightG(const QVector<  QVector < ct::Matf > > &W);
	void set_weightB(const QVector<  QVector < ct::Matf > > &W);

private:
	Ui::DrawCnvWeight *ui;

public slots:
	void onTimeout();

	// QWidget interface
protected:
	virtual void paintEvent(QPaintEvent *event);

private:
	QVector< QVector < ct::Matf > > m_W_R, m_prevW_R;
	QVector< QVector < ct::Matf > > m_W_G, m_prevW_G;
	QVector< QVector < ct::Matf > > m_W_B, m_prevW_B;
	QPoint m_pt;
	int m_offset;
	bool m_update;
	QTimer m_timer;

	QSize draw_weight(QPainter& painter, int offset);

	// QWidget interface
protected:
	void mouseReleaseEvent(QMouseEvent *event);
	void mouseDoubleClickEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
	void mousePressEvent(QMouseEvent *event);

};

#endif // DRAWCNVWEIGHT_H
