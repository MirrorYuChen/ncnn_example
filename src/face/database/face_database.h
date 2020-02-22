#ifndef _FACE_DATABASE_H_
#define _FACE_DATABASE_H_

#include <map>
#include <memory>
#include <vector>

#include "opencv2/core.hpp"
#include "./stream/file_stream.h"
#include "../../common/common.h"

namespace mirror {

class FaceDatabase {
public:
	FaceDatabase();
	~FaceDatabase();

	bool Save(const char* path) const;
	bool Load(const char* path);
	int64_t Insert(const std::vector<float>& feat, const std::string& name);
	int Delete(const std::string& name);
	int64_t QueryTop(const std::vector<float>& feat, QueryResult* query_result = nullptr);

	void Clear();

private:
	FaceDatabase(const FaceDatabase &other) = delete;
	const FaceDatabase &operator=(const FaceDatabase &other) = delete;

private:
	class Impl;
	Impl* impl_;

};
}


#endif // !_FACE_DATABASE_H_

