#include "widgetcifar.h"
#include "ui_widgetcifar.h"

#include <QPainter>
#include <QPaintEvent>
#include <QImage>
#include <QVariant>
#include <QMap>
#include <QTime>

#include <time.h>
#include <random>

const int wim		= 32;
const int him		= 32;

////////////////////////

WidgetCifar::WidgetCifar(QWidget *parent) :
	QWidget(parent),
	ui(new Ui::WidgetCifar)
{
	ui->setupUi(this);

	m_index = 0;
	m_cifar = 0;
	m_mode = TRAIN;

	m_start_show = 0;

	m_timer.setSingleShot(true);
	m_timer.setInterval(300);
	connect(&m_timer, SIGNAL(timeout()), this, SLOT(onTimeout()));

	m_cols = 0;

	m_last_test_pos = 0;
	m_last_test_pos_saved = 0;

	m_update = false;
	m_timer_update.start(30);
	connect(&m_timer_update, SIGNAL(timeout()), this, SLOT(onTimeoutUpdate()));

	setMouseTracking(true);
}

WidgetCifar::~WidgetCifar()
{
	delete ui;
}

void WidgetCifar::setTestMode()
{
	m_mode = TEST;
	m_timer.start();
}

void WidgetCifar::setTrainMode()
{
	m_mode = TRAIN;
	m_timer.start();
}

int WidgetCifar::mode() const
{
	return m_mode;
}

void WidgetCifar::setCifar(cifar_reader *val)
{
	m_cifar = val;
	update_source();
}

double WidgetCifar::index() const
{
	return m_index;
}

void WidgetCifar::updatePredictfromIndex(const QVector<int> &predict, int index)
{
	if(!m_cifar || !m_cifar->isBinDataExists())
		return;

	QVector< int >& prediction = m_mode == TRAIN? m_prediction_train : m_prediction_test;

	qDebug("predicted array size %d", predict.size());

	if(predict.size() + index > prediction.size()){
		prediction.resize(predict.size() + index);
		prediction.fill(0);
	}
#pragma omp parallel for
	for(int i = 0; i < predict.size(); i++){
		prediction[i + index] = predict[i];
	}
	update();
}

void WidgetCifar::next()
{
	if(!m_cifar || !m_cifar->isBinDataExists())
		return;
	if(m_mode == TRAIN){
		if(m_index < 1){
			m_index = m_cifar->current_percent();
		}
		update_source();
	}else{
		m_last_test_pos += m_last_test_pos_saved;
		update_test();
	}
}

void WidgetCifar::prev()
{
	if(m_cifar->isBinDataExists())
		toBegin();
}

void WidgetCifar::toBegin()
{
	if(m_mode == TRAIN){
		if(m_cifar->isBinDataExists()){
			m_index = 0;
			update_source();
		}
	}else{
		m_last_test_pos = 0;
		m_last_test_pos_saved = 0;
		update_test();
	}
}

size_t WidgetCifar::count() const
{
	return m_output_data.size();
}

const QVector<TData> &WidgetCifar::output_data() const
{
	return m_output_data;
}

int WidgetCifar::test_pos() const
{
	return m_last_test_pos;
}

void WidgetCifar::onTimeout()
{
	if(m_mode == TRAIN){
		update_source();
	}else{
		update_test();
	}
}

void WidgetCifar::onTimeoutUpdate()
{
	if(m_update){
		m_update = false;
		int xid = m_point.x() / cifar_reader::WidthIM;
		int yid = m_point.y() / cifar_reader::HeightIM;
		int off = yid * m_cols + xid;

		if(off < m_output_data.size()){
			QImage im((uchar*)m_output_data[off].image.data(), wim, him, QImage::Format_RGB888);
			m_sel_image = im;
			update();
		}
	}

	if(QTime::currentTime().msecsSinceStartOfDay() - m_start_show  > 10000
			&& !m_sel_image.isNull()){
		m_sel_image = QImage();
		update();
	}
}

void WidgetCifar::update_source()
{
	if(!m_cifar || !m_cifar->isBinDataExists())
		return;

	bool floop = true;

	int x = 0, y = 0;
	int off = 0, batch = 100;

	int wim = cifar_reader::WidthIM;
	int him = cifar_reader::HeightIM;

	m_output_data.clear();
	double cp = m_index;

	while(floop){

		QVector< TData >& data = m_cifar->train(batch, cp);
		cp = m_cifar->current_percent();

		m_output_data.append(data);
		off += batch;

		for(int i = 0; i < data.size(); i++){
			if(y * him + him >= height()){
				floop = false;
				continue;
			}

			x++;

			if((x + 1) * wim > width()){
				m_cols = x;
				x = 0;
				y++;
			}
		}
	}
	update();
}

void WidgetCifar::update_test()
{
	if(!m_cifar || !m_cifar->isBinDataExists())
		return;

	bool floop = true;

	int x = 0, y = 0;
	int cnt = 100;

	int wim = cifar_reader::WidthIM;
	int him = cifar_reader::HeightIM;

	m_output_data.clear();
	int beg = m_last_test_pos;

	while(floop){

		QVector< TData >& data = m_cifar->test(beg, cnt);
		cnt = data.size();

		m_output_data.append(data);
		beg += cnt;

		for(int i = 0; i < data.size(); i++){
			if(y * him + him >= height()){
				floop = false;
				continue;
			}

			x++;

			if((x + 1) * wim > width()){
				m_cols = x;
				x = 0;
				y++;
			}
		}
	}
	m_last_test_pos_saved = beg;

	update();
}

//void rotate_mnist(const QByteArray& in, QByteArray& out, int w, int h, float angle)
//{
//	if(in.size() != w * h)
//		return;

//	float cw = w / 2;
//	float ch = h / 2;

//	out.resize(in.size());
//	out.fill(0);

//	for(int y = 0; y < h; y++){
//		for(int x = 0; x < w; x++){
//			float x1 = x - cw;
//			float y1 = y - ch;

//			float nx = x1 * cos(angle) + y1 * sin(angle);
//			float ny = -x1 * sin(angle) + y1 * cos(angle);
//			nx += cw; ny += ch;
//			int ix = nx, iy = ny;
//			if(ix >= 0 && ix < w && iy >= 0 && iy < h){
//				char c = in[y * w + x];
//				out[iy * w + ix] = c;
//			}
//		}
//	}
//}

#define PI	3.1415926535897932384626433832795

void WidgetCifar::paintEvent(QPaintEvent *event)
{
	Q_UNUSED(event);

	QPainter painter(this);

	if(!m_cifar)
		return;

	painter.setBrush(Qt::NoBrush);

	int x = 0, y = 0;

	int wim = cifar_reader::WidthIM;
	int him = cifar_reader::HeightIM;

	QVector< int >& prediction = m_mode == TRAIN? m_prediction_train : m_prediction_test;

	for(int i = 0; i < m_output_data.size(); i++){
		if(y * him + him >= height()){
			continue;
		}

		QByteArray _out;
		_out = m_output_data[i].image;

		QImage im((uchar*)_out.data(), wim, him, QImage::Format_RGB888);

		painter.setPen(Qt::red);
		painter.drawImage(x * wim, y * him, im);
		painter.drawRect(x * wim, y * him, wim, him);

		painter.setPen(Qt::green);
		QString text = QString::number((uint)m_output_data[i].lb);
		painter.drawText(x * wim + 3, y * him + 12, text);

		if(prediction.size() && prediction.size() >= m_output_data.size()){
			painter.setPen(QColor(30, 255, 100));
			QString text = QString::number((uint)prediction[i]);
			painter.drawText(x * wim + 17, y * him + 12, text);

			if(m_output_data[i].lb != prediction[i]){
				QPen pen;
				pen.setColor(Qt::yellow);
				pen.setWidth(2);
				painter.setBrush(Qt::NoBrush);
				painter.setPen(pen);
				painter.drawRect(x * wim + 2, y * him + 2, wim - 4, him - 4);
			}else{
				QPen pen;
				pen.setColor(Qt::green);
				pen.setWidth(2);
				painter.setBrush(Qt::NoBrush);
				painter.setPen(pen);
				painter.drawRect(x * wim + 2, y * him + 2, wim - 4, him - 4);
			}
		}

		x++;

		if((x + 1) * wim > width()){
			x = 0;
			y++;
		}
	}

	if(!m_sel_image.isNull()){
		QImage tmp = m_sel_image.scaled(QSize(100, 100));
		painter.drawImage(m_point, tmp);
	}
}

void WidgetCifar::resizeEvent(QResizeEvent *event)
{
	m_timer.start();
}


void WidgetCifar::mouseMoveEvent(QMouseEvent *event)
{
	m_point = event->pos();
	m_update = true;

	m_start_show = QTime::currentTime().msecsSinceStartOfDay();
}
