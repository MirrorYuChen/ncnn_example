#ifndef _FILE_STREAM_H_
#define _FILE_STREAM_H_

#include <fstream>

namespace mirror {

class StreamWriter {
public:
	using self = StreamWriter;
	StreamWriter() {}
	virtual ~StreamWriter() = default;
	virtual size_t write(const char* data, size_t length) = 0;

};

class StreamReader {
public:
	using self = StreamReader;
	StreamReader() {}
	virtual ~StreamReader() = default;
	virtual size_t read(char* data, size_t length) = 0;
};

class FileStream : public StreamWriter, public StreamReader {
public:
	enum Mode {
		Input  = 0x01,
		Output = 0x01 << 1,
		Binary = 0x01 << 2,
	};

	FileStream() {}
	explicit FileStream(const std::string &path, int mode = Output) {
		open(path, mode);
	}

	FileStream(FileStream &&other) {
		std::swap(iofile_, other.iofile_);
	}

	const FileStream &operator=(FileStream &&other) {
		std::swap(iofile_, other.iofile_);
		return *this;
	}
	virtual ~FileStream() {
		close();
	}

	bool open(const std::string &path, int mode = Output);
	void close();
	bool is_opened() const;
	size_t write(const char *data, size_t length) override;
	size_t read(char *data, size_t length) override;

private:
	FileStream(const FileStream& other) = delete;
	FILE* iofile_ = nullptr;

};

class FileWriter : public FileStream {
public:
	using self = FileWriter;
	using supper = FileStream;
	FileWriter() {};
	explicit FileWriter(const std::string& path, int mode = Output)
		: FileStream(path, (mode & (~Input)) | Output) {
	}
	~FileWriter() {};

	bool open(const std::string& path, int mode = Output) {
		return supper::open(path, (mode & (!Input)) | Output);
	}

};

class FileReader : public FileStream {
public:
	using self = FileReader;
	using supper = FileStream;

	FileReader() {}
	explicit FileReader(const std::string& path, int mode = Input)
		: FileStream(path, (mode & (~Output)) | Input) {

	}

	bool open(const std::string& path, int mode = Input) {
		return supper::open(path, (mode & (~Output)) | Input);
	}


};


// read and write value
template <typename T>
static size_t Write(StreamWriter &writer, const T &value) {
	return writer.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

template <typename T>
static size_t Read(StreamReader &reader, T &value) {
	return reader.read(reinterpret_cast<char *>(&value), sizeof(T));
}

// read and write array
template <typename T>
static size_t Write(StreamWriter &writer, const T *arr, size_t size) {
	return writer.write(reinterpret_cast<const char *>(arr), sizeof(T) * size);
}

template <typename T>
static size_t Read(StreamReader &reader, T *arr, size_t size) {
	return reader.read(reinterpret_cast<char *>(arr), sizeof(T) * size);
}



}






#endif // !_FILE_STREAM_H_

