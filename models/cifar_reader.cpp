#include "cifar_reader.h"

#include <QFile>
#include <QDir>
#include <QSettings>

#include <random>

const QString train_images_file[] = {
	("data_batch_1.bin"),
	("data_batch_2.bin"),
	("data_batch_3.bin"),
	("data_batch_4.bin"),
	("data_batch_5.bin")
};

const QString test_images_file = "test_batch.bin";

const int rowLen[] = {
	3073,
	3074
};

const int outLen[] = {
	10,
	100
};

const int channels = 3;

const int countFiles = sizeof(train_images_file) / sizeof(*train_images_file);

const int sizeCData = 3072;
const int countBin = 10000;
const int maxCount = countFiles * countBin;

union ULab{
	ULab(){
		sh = 0;
	}

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
	m_count_test = 0;
	m_directory = "data/cifar-10-batches-bin";

	loadSettings();

	m_timer.setSingleShot(true);
	m_timer.setInterval(10000);
	connect(&m_timer, SIGNAL(timeout()), this, SLOT(onTimeout()));

	m_timer_test.setSingleShot(true);
	m_timer_test.setInterval(10000);
	connect(&m_timer_test, SIGNAL(timeout()), this, SLOT(onTimeoutTest()));
}

cifar_reader::~cifar_reader()
{
	saveSettings();
}

QVector<TData> &cifar_reader::train(int batch, double percent)
{
	m_current_data.clear();
	if(batch == 0 || percent > 1)
		return m_current_data;

	int num = countFiles * percent;
	double offset = percent - (num * 1./countFiles);
	offset *= maxCount;

	if(offset + batch > countBin){
		batch = countBin - offset;
	}

	QString fn = m_directory + "/" + train_images_file[(int)num];

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

QVector<TData> &cifar_reader::test(int beg, int count)
{
	m_current_test.clear();

	open_test_file();
	if(!m_current_test_object.isOpen())
		throw new std::invalid_argument("cifar_reader::test: test file not opened");
	readCifar(m_current_test_object, m_current_test, count, beg);
	m_timer_test.start();
	return m_current_test;
}

uint cifar_reader::count_test()
{
	open_test_file();
	if(!m_current_test_object.isOpen()){
		return 0;
	}
	if(!m_count_test){
		m_count_test = m_current_test_object.size() / rowLen[m_data_source];
		m_timer_test.start();
	}
	return m_count_test;
}

void cifar_reader::open_test_file()
{
	if(!m_current_test_object.isOpen()){
		m_current_test_object.setFileName(m_directory + "/" + test_images_file);
		m_current_test_object.open(QIODevice::ReadOnly);
		qDebug("Test file open: %d, error: %s", m_current_test_object.isOpen(), m_current_test_object.errorString().toLatin1().data());
	}
}

void cifar_reader::convToXy(const QVector<TData> &data, int first, int last, std::vector<ct::Matf> &X, ct::Matf *y)
{
	if(data.empty() || first >= last)
		return;

	int size = last - first;

	if(y){
		y->setSize(size, 1);
		y->fill(0);
		float *dy = y->ptr();
		int i = 0;
		for(int i = first, j = 0; i < last; ++i, ++j){
			dy[j * y->cols + 0] = data[i].lb;
		}
	}

	X.resize(channels);

	for(size_t i = 0; i < X.size(); ++i){
		X[i].setSize(size, WidthIM * HeightIM);
	}

	for(int i = first, j = 0; i < last; ++i, ++j){
		ct::image2mats(data[i].data, WidthIM, HeightIM, j, X[0], X[1], X[2]/*, X[3]*/);
	}
}

bool cifar_reader::getData(int file, int offset, TData &data)
{
	QString fn = m_directory + "/" + train_images_file[(int)file];

	if(m_current_file == file
			&& m_current_offset == offset){
		return false;
	}

	m_current_file = file;
	m_current_offset = offset;

	//qDebug("next num file %d %f", num, offset);

	readCifar1(fn, data, offset);

	return true;
}

bool cifar_reader::getData(double percent, TData &data)
{
	int num = countFiles * percent;
	double offset = percent - (num * 1./countFiles);
	offset *= maxCount;

	QString fn = m_directory + "/" + train_images_file[(int)num];

	if(m_current_file == num
			&& m_current_offset == offset){
		return false;
	}

	m_current_file = num;
	m_current_offset = offset;

	//qDebug("next num file %d %f", num, offset);

	readCifar1(fn, data, offset);

	return true;
}

void cifar_reader::getTrain(int batch, std::vector<ct::Matf> &X, ct::Matf &y)
{
	std::uniform_real_distribution<double> urnd(0, 1);

	X.resize(channels);

	y.setSize(batch, 1);
	y.fill(0);

	for(size_t i = 0; i < X.size(); ++i){
		X[i].setSize(batch, WidthIM * HeightIM);
	}

	float *dy = y.ptr();

	std::map< int, std::map<int, bool> > values_exists;
	std::vector< ct::Vec2i > vals;

	for(int i = 0; i < batch; ++i){
		bool next = true;
		do{
			double val = urnd(generator);
			int file, offset;
			get_file_offset(val, file, offset);
			if(values_exists.find(file) == values_exists.end()){
				values_exists[file][offset] = true;
			}else{
				std::map< int, bool >& vb = values_exists[file];
				if(vb.find(offset) == vb.end()){
					values_exists[file][offset] = true;
					vals.push_back(ct::Vec2i(file, offset));
					next = false;
				}
			}
		}while(next);
	}

	for(int i = 0; i < batch; ++i){
		TData data;
		ct::Vec2i& v = vals[i];

		getData(v[0], v[1], data);

		if((data.lb == 3 || data.lb == 5 ||data.lb == 2) && i < batch - 1){
			ct::image2mats(data.data, WidthIM, HeightIM, i, X[0], X[1], X[2]);
			dy[i * y.cols + 0] = data.lb;
			i++;
		}
		ct::image2mats(data.data, WidthIM, HeightIM, i, X[0], X[1], X[2]/*, X[3]*/);
		dy[i * y.cols + 0] = data.lb;
	}
}

void cifar_reader::getTrain2(int batch, std::vector<ct::Matf> &X, ct::Matf &y)
{
	std::uniform_real_distribution<double> urnd(0, 1);

	X.resize(batch);

	y.setSize(batch, 1);
	y.fill(0);

	float *dy = y.ptr();

	std::map< int, std::map<int, bool> > values_exists;
	std::vector< ct::Vec2i > vals;

	for(int i = 0; i < batch; ++i){
		bool next = true;
		do{
			double val = urnd(generator);
			int file, offset;
			get_file_offset(val, file, offset);
			if(values_exists.find(file) == values_exists.end()){
				values_exists[file][offset] = true;
			}else{
				std::map< int, bool >& vb = values_exists[file];
				if(vb.find(offset) == vb.end()){
					values_exists[file][offset] = true;
					vals.push_back(ct::Vec2i(file, offset));
					next = false;
				}
			}
		}while(next);
	}

	for(int i = 0; i < batch; ++i){
		TData data;
		ct::Vec2i& v = vals[i];

		getData(v[0], v[1], data);

		ct::image2mat(data.data, WidthIM, HeightIM, X[i]);
		dy[i * y.cols + 0] = data.lb;
	}

}

bool cifar_reader::getDataIt(double percent, int batch, QVector< TData > &data)
{
	int num = countFiles * percent;
	double offset = percent - (num * 1./countFiles);
	offset *= maxCount;

	QString fn = m_directory + "/" + train_images_file[(int)num];

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
	X.resize(channels);

	for(size_t i = 0; i < X.size(); ++i){
		X[i].setSize(batch, WidthIM * HeightIM);
	}

	QVector< TData > data;
	getDataIt(percent, batch, data);

	if(y){
		y->setSize(batch, 1);
		y->fill(0);
		float *dy = y->ptr();
		for(int i = 0; i < batch; ++i){
			dy[i * y->cols + 0] = data[i].lb;
		}
	}


	for(int i = 0; i < batch; ++i){
		ct::image2mats(data[i].data, WidthIM, HeightIM, i, X[0], X[1], X[2]/*, X[3]*/);
	}
}

uint cifar_reader::getTest(uint beg, uint batch, std::vector<ct::Matf> &Xs, ct::Matf &y)
{
	test(beg, batch);

	if(m_current_test.empty())
		return 0;

	Xs.resize(channels);

	y.setSize(m_current_test.size(), 1);
	y.fill(0);

	for(size_t i = 0; i < Xs.size(); ++i){
		Xs[i].setSize(m_current_test.size(), WidthIM * HeightIM);
	}

	float *dy = y.ptr();

	for(uint i = 0; i < m_current_test.size(); ++i){
		TData &data = m_current_test[i];

		ct::image2mats(data.data, WidthIM, HeightIM, i, Xs[0], Xs[1], Xs[2]/*, Xs[3]*/);

		dy[i * y.cols + 0] = data.lb;
	}
	return m_current_test.size();
}

uint cifar_reader::getTest(uint batch, std::vector<ct::Matf> &Xs, ct::Matf &y)
{
	if(batch <= 0)
		return 0;

	open_test_file();

	std::uniform_int_distribution<int> urnd(0, countBin);

	m_current_test.resize(batch);
	for(int i = 0; i < batch; ++i){
		int off = urnd(ct::generator);
		readCifar1(m_current_test_object, m_current_test[i], off);
	}

	Xs.resize(channels);

	y.setSize(m_current_test.size(), 1);
	y.fill(0);

	for(size_t i = 0; i < Xs.size(); ++i){
		Xs[i].setSize(m_current_test.size(), WidthIM * HeightIM);
	}

	float *dy = y.ptr();

	for(uint i = 0; i < m_current_test.size(); ++i){
		TData &data = m_current_test[i];

		ct::image2mats(data.data, WidthIM, HeightIM, i, Xs[0], Xs[1], Xs[2]/*, Xs[3]*/);

		dy[i * y.cols + 0] = data.lb;
	}
	return m_current_test.size();
}

uint cifar_reader::getTest2(uint batch, std::vector<ct::Matf> &Xs, ct::Matf &y)
{
	if(batch <= 0)
		return 0;

	open_test_file();

	std::uniform_int_distribution<int> urnd(0, countBin);

	std::map<int, bool > exists;

	m_current_test.resize(batch);
	for(int i = 0; i < batch; ++i){
		int off = 0;
		do{
			off = urnd(ct::generator);
		}while(exists[off]);

		exists[off] = true;

		readCifar1(m_current_test_object, m_current_test[i], off);
	}

	Xs.resize(batch);

	y.setSize(m_current_test.size(), 1);
	y.fill(0);

	float *dy = y.ptr();

	for(uint i = 0; i < m_current_test.size(); ++i){
		TData &data = m_current_test[i];

		ct::image2mat(data.data, WidthIM, HeightIM, Xs[i]/*, Xs[3]*/);

		dy[i * y.cols + 0] = data.lb;
	}
	return m_current_test.size();
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

bool cifar_reader::openDir(const QString &dir)
{
	m_directory = dir;

	if(isBinDataExists()){
		qDebug("binary data exists");
	}else{
		qDebug("binary data not exists");
	}
	return isBinDataExists();
}

QString cifar_reader::currentDirectory() const
{
	return m_directory;
}

bool cifar_reader::isBinDataExists() const
{
	bool ret = true;

	for(int i = 0; i < countFiles; ++i){
		ret &= QFile::exists(m_directory + "/" + train_images_file[i]);
	}
	ret &= QFile::exists(m_directory + "/" + test_images_file);

	return ret;
}

void cifar_reader::onTimeout()
{
	if(m_current_object.isOpen())
		m_current_object.close();

}

void cifar_reader::onTimeoutTest()
{
	if(m_current_test_object.isOpen())
		m_current_test_object.close();
}

void cifar_reader::get_file_offset(double percent, int &file, int &offset)
{
	int num = countFiles * percent;
	double _offset = percent - (num * 1./countFiles);
	_offset *= maxCount;

	file = num;
	offset = _offset;
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
	if(!QFile::exists(fn))
		throw new std::invalid_argument("cifar_reader::readCifar file not exists");

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
	//max_off -= 1;
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

void cifar_reader::loadSettings()
{
	QSettings settings("config.ini", QSettings::IniFormat);
	settings.beginGroup("main");
	QString tmp = settings.value("directory").toString();
	settings.endGroup();

	if(!tmp.isEmpty())
		m_directory = tmp;
}

void cifar_reader::saveSettings()
{
	if(!isBinDataExists()){
		return;
	}
	QSettings settings("config.ini", QSettings::IniFormat);
	settings.beginGroup("main");
	settings.setValue("directory", m_directory);
	settings.endGroup();
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
