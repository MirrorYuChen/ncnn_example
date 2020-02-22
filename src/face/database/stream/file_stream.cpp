#include "file_stream.h"

namespace mirror {
bool FileStream::open(const std::string & path, int mode) {
	close();
	std::string mode_str;
	if ((mode & Input) && (mode & Output)) {
		mode_str += "a+";
	} else {
		if (mode & Input) {
			mode_str += "r";
		} else {
			mode_str += "w";
		}
	}
	if (mode & Binary) mode_str += "b";
#if _MSC_VER >= 1600
	fopen_s(&iofile_, path.c_str(), mode_str.c_str());
#else
	iofile_ = std::fopen(path.c_str(), mode_str.c_str());
#endif
	return iofile_ != nullptr;
}

void FileStream::close() {
	if (iofile_ != nullptr) std::fclose(iofile_);
}

bool FileStream::is_opened() const {
	return iofile_ != nullptr;
}

size_t FileStream::write(const char* data, size_t length) {
	if (iofile_ == nullptr) return 0;
	auto result = std::fwrite(data, 1, length, iofile_);
	return size_t(result);
}

size_t FileStream::read(char* data, size_t length) {
	if (iofile_ == nullptr) return 0;
	auto result = std::fread(data, 1, length, iofile_);
	return size_t(result);
}


}


