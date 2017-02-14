#include "cifar_reader.h"

#include <QFile>
#include <QDir>

#include <random>

const QString train_images_file[] = {
	("data/cifar-10-batches-bin/data_batch_1.bin"),
	("data/cifar-10-batches-bin/data_batch_2.bin"),
	("data/cifar-10-batches-bin/data_batch_3.bin"),
	("data/cifar-10-batches-bin/data_batch_4.bin"),
	("data/cifar-10-batches-bin/data_batch_5.bin")
};

const int rowLen[] = {
	3073,
	3074
};

const int outLen[] = {
	10,
	100
};

const int countFiles = sizeof(train_images_file) / sizeof(*train_images_file);

const int sizeCData = 3072;
const int countBin = 10000;
const int maxCount = countFiles * countBin;

union ULab{
	uchar b1;
	uchar b2[2];
	ushort sh;
};

std::mt19937 generator;

/////////////////////////

inline uint invert(uint val)
{
	return (val >> 24) | (((val >> 16) & 0xFF) << 8) |
			(((val >> 8) & 0xFF) << 16) | (val << 24);
}

/////////////////////////

cifar_reader::cifar_reader()
{
	m_data_source = CIFAR10;
	m_count = 0;
	m_current_file = 0;
	m_current_batch = 0;
	m_current_offset = 0;
	m_current_percent = 0;

	m_timer.setSingleShot(true);
	m_timer.setInterval(10000);
	connect(&m_timer, SIGNAL(timeout()), this, SLOT(onTimeout()));
}

QVector<TData> &cifar_reader::train(int batch, double percent)
{
	if(batch == 0 || percent > 1)
		return m_current_data;

	int num = countFiles * percent;
	double offset = percent - (num * 1./countFiles);
	offset *= maxCount;

	if(offset + batch > countBin){
		batch = countBin - offset;
	}

	QString fn = train_images_file[(int)num];

	if(m_current_file != num){
		qDebug("next num file %d", num);
	}

	if(m_current_batch == batch
			&& m_current_file == num
			&& m_current_offset == offset
			&& !m_current_data.isEmpty()){
		return m_current_data;
	}

	m_current_batch = batch;
	m_current_file = num;
	m_current_offset = offset;

	m_current_percent = (double)(num * countBin + m_current_offset + batch) / (maxCount);

	m_current_data.clear();
	readCifar(fn, m_current_data, batch, offset);

	return m_current_data;
}

bool cifar_reader::getData(double percent, TData &data)
{
	int num = countFiles * percent;
	double offset = percent - (num * 1./countFiles);
	offset *= maxCount;

	QString fn = train_images_file[(int)num];

	if(m_current_file == num
			&& m_current_offset == offset){
		return false;
	}

	m_current_file = num;
	m_current_offset = offset;

	//qDebug("next num file %d", num);

	readCifar1(fn, data, offset);

	return true;
}

void cifar_reader::getTrain(int batch, std::vector<ct::Matf> &X, ct::Matf &y)
{
	std::uniform_real_distribution<double> urnd(0, 1);

	X.resize(3);

	y.setSize(batch, outLen[m_data_source]);
	y.fill(0);

	for(size_t i = 0; i < X.size(); ++i){
		X[i].setSize(batch, WidthIM * HeightIM);
	}


	float *dy = y.ptr();

	for(int i = 0; i < batch; ++i){
		TData data;
		while(!getData(urnd(generator), data)){};

		ct::image2mats(data.data, WidthIM, HeightIM, i, X[0], X[1], X[2]);

		dy[i * y.cols + data.lb] = 1.f;
	}
}

bool cifar_reader::getDataIt(double percent, int batch, QVector< TData > &data)
{
	int num = countFiles * percent;
	double offset = percent - (num * 1./countFiles);
	offset *= maxCount;

	QString fn = train_images_file[(int)num];

	if(m_current_file == num
			&& m_current_offset == offset){
		return false;
	}

	m_current_file = num;
	m_current_offset = offset;

	//qDebug("next num file %d", num);

	readCifar(fn, data, batch, offset);

	return true;
}

void cifar_reader::getTrainIt(double percent, int batch, std::vector<ct::Matf> &X, ct::Matf *y)
{
	X.resize(3);

	for(size_t i = 0; i < X.size(); ++i){
		X[i].setSize(batch, WidthIM * HeightIM);
	}

	QVector< TData > data;
	getDataIt(percent, batch, data);

	if(y){
		y->setSize(batch, outLen[m_data_source]);
		y->fill(0);
		float *dy = y->ptr();
		for(int i = 0; i < batch; ++i){
			dy[i * y->cols + data[i].lb] = 1.f;
		}
	}


	for(int i = 0; i < batch; ++i){
		ct::image2mats(data[i].data, WidthIM, HeightIM, i, X[0], X[1], X[2]);
	}
}

uint cifar_reader::count()
{
	return maxCount;
}

uint cifar_reader::current_file()
{
	return m_current_file;
}

uint cifar_reader::max_files() const
{
	return countFiles;
}

double cifar_reader::current_percent() const
{
	return m_current_percent;
}

void cifar_reader::onTimeout()
{
	if(m_current_object.isOpen())
		m_current_object.close();
}

//ct::Matf &cifar_reader::X()
//{
//	return m_X;
//}

//ct::Matf &cifar_reader::y()
//{
//	return m_y;
//}

uint cifar_reader::readCifar(const QString& fn, QVector<TData> &val, int batch, int offset)
{
	if(m_current_object.fileName() == fn){
		if(!m_current_object.isOpen()){
			m_current_object.open(QIODevice::ReadOnly);
		}
		if(m_current_object.isOpen()){
			uint cnt = readCifar(m_current_object, val, batch, offset);
			m_timer.start();
			return cnt;
		}
	}else{
		m_current_object.close();
		m_current_object.setFileName(fn);
		if(m_current_object.open(QIODevice::ReadOnly)){
			uint cnt = readCifar(m_current_object, val, batch, offset);
			m_timer.start();
			return cnt;
		}
	}
	return 0;
}

namespace labels{

const int lbSz[] = {1, 2};

inline ushort getLb(ULab lb, int source)
{
	switch (source) {
		case cifar_reader::CIFAR10:
			return lb.b1;
		case cifar_reader::CIFAR100:
			return lb.sh;
	}
}

}

uint cifar_reader::readCifar(QFile &file, QVector<TData> &val, int batch, int offset)
{
	if(!file.isOpen())
		return 0;

	int sz = 0;

	int max_off = file.size() / rowLen[m_data_source];
	max_off -= 1;
	m_count = max_off;

	sz = std::min(max_off - offset, batch);

	file.seek(offset * rowLen[m_data_source]);
	qDebug("fp: %d", file.pos());

	val.resize(sz);

	for(int i = 0; i < sz; ++i){
		val[i].data.resize(sizeCData);

		ULab lb;
		file.read((char*)lb.b2, labels::lbSz[m_data_source]);
		file.read(val[i].data.data(), sizeCData);

		val[i].toImage();

		val[i].lb = labels::getLb(lb, m_data_source);
	}


	return sz;
}

uint cifar_reader::readCifar1(const QString &fn, TData &val, int offset)
{
	if(m_current_object.fileName() == fn){
		if(!m_current_object.isOpen()){
			m_current_object.open(QIODevice::ReadOnly);
		}
		if(m_current_object.isOpen()){
			uint cnt = readCifar1(m_current_object, val, offset);
			m_timer.start();
			return cnt;
		}
	}else{
		m_current_object.close();
		m_current_object.setFileName(fn);
		if(m_current_object.open(QIODevice::ReadOnly)){
			uint cnt = readCifar1(m_current_object, val, offset);
			m_timer.start();
			return cnt;
		}
	}
	return 0;
}

uint cifar_reader::readCifar1(QFile &file, TData &val, int offset)
{
	if(!file.isOpen())
		return 0;

	if(offset >= 0){
		file.seek(offset * rowLen[m_data_source]);

		val.data.resize(sizeCData);

		ULab lb;
		file.read((char*)lb.b2, labels::lbSz[m_data_source]);
		file.read(val.data.data(), sizeCData);

		val.lb = labels::getLb(lb, m_data_source);
	}
	return !val.data.isEmpty();
}

///******************************

void TData::toImage()
{
	if(!image.isNull() || data.isEmpty())
		return;

	const int w = cifar_reader::WidthIM;
	const int h = cifar_reader::HeightIM;

	image.resize(w * h * 3);

	const int bpp = 3;

	uchar *id = (uchar*)image.data();
	uchar *od = (uchar*)data.data();
	for(int y = 0; y < h; ++y){
		for(int x = 0; x < w; ++x){
			id[bpp * (y * w + x)]		= od[y * w + x];
			id[bpp * (y * w + x) + 1]	= od[w * h + y * w + x];
			id[bpp * (y * w + x) + 2]	= od[2 * w * h + y * w + x];
		}
	}
}
