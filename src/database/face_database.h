#ifndef _FACE_DATABASE_H_
#define _FACE_DATABASE_H_

#include <map>
#include <memory>
#include <vector>

#include "file_stream.h"
#include "opencv2/core.hpp"
#include "../common/common.h"

#define kFaceFeatureDim 128
#define kFaceNameDim 128

class FaceDatabase {
public:
	FaceDatabase();
	~FaceDatabase();

	bool Save(const char *path) const;
	bool Load(const char *path);
	int64_t Insert(const std::vector<float>& feat, const std::string& name);
	int Delete(int64_t index);
	int64_t QueryTop(const std::vector<float>& feat, QueryResult *query_result = nullptr);


	size_t Count() const;
	void Clear();

private:
	FaceDatabase(const FaceDatabase &other) = delete;
	const FaceDatabase &operator=(const FaceDatabase &other) = delete;

private:
	class Impl;
	Impl* impl_;

};


#endif // !_FACE_DATABASE_H_

