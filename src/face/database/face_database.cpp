#include "face_database.h"
#include <iostream>

namespace mirror {

class FaceDatabase::Impl {
public:
	Impl() {
		max_index_ = 0;
		features_db_.clear();
	}

	~Impl() {

	}
	
	bool Save(StreamWriter& writer) const {
		const uint64_t num_faces = db_.size();
		const uint64_t dim_feat = kFaceFeatureDim;
		const uint64_t dim_name = kFaceNameDim;

		Write(writer, num_faces);
		for (auto& line : db_) {
			auto& name = line.first;
			auto& feat = line.second;

			char name_arr[kFaceNameDim];
			sprintf(name_arr, "%s", name.c_str());

			Write(writer, name_arr, size_t(dim_name));
			Write(writer, &feat[0], size_t(dim_feat));
			
		}

		std::cout << "FaceDatabase Loaded " << num_faces << " faces" << std::endl;
		return true;
	}

	bool Load(StreamReader& reader) {
		uint64_t num_faces = 0;
		const uint64_t dim_feat = kFaceFeatureDim;
		const uint64_t dim_name = kFaceNameDim;

		Read(reader, num_faces);
		std::cout << "number faces is: " << num_faces << std::endl;

		db_.clear();
		max_index_ = -1;

		for (size_t i = 0; i < num_faces; ++i) {
			char name_arr[kFaceNameDim];
			Read(reader, name_arr, size_t(dim_name));
			std::cout << "name is: " << name_arr << std::endl;

			std::vector<float> feat(kFaceFeatureDim);
			Read(reader, &feat[0], size_t(dim_feat));

			db_.insert(std::make_pair(std::string(name_arr), feat));
			max_index_ = (max_index_ > i ? max_index_ : i);
		}
		++max_index_;

		std::cout << "FaceDatabase Loaded " << num_faces << " faces" << std::endl;

		return true;
	}


	int64_t Insert(const std::string& name, const std::vector<float>& feat) {
		int64_t new_index = max_index_++;
		std::cout << "new index is: " << new_index << std::endl;
		db_.insert(std::make_pair(name, feat));
		return new_index;
	}

	int Delete(const std::string& name) {
		std::map<std::string, std::vector<float>>::iterator it = db_.find(name);
		if (it != db_.end()) {
			db_.erase(it);
			std::cout << "Delete: " << name << " successed." << std::endl;
		}
		return 0;
	}

	void Clear() {
		db_.clear();
		max_index_ = 0;
	}



	float CalculateSimilarity(const std::vector<float>& feat1,
		const std::vector<float>& feat2) {
		double dot = 0;
		double norm1 = 0;
		double norm2 = 0;
		for (size_t i = 0; i < kFaceFeatureDim; ++i) {
			dot += feat1[i] * feat2[i];
			norm1 += feat1[i] * feat1[i];
			norm2 += feat2[i] * feat2[i];
		}

		return dot / (sqrt(norm1 * norm2) + 1e-5);
	}

	bool Compare(const std::vector<float>& feat1,
		const std::vector<float>& feat2, float *similarity) {
		if (feat1.size() == 0 || feat2.size() == 0 || !similarity) return false;
		*similarity = CalculateSimilarity(feat1, feat2);
		return true;
	}

	size_t QueryTop(const std::vector<float>& feat,
		QueryResult *query_result) {
		std::vector<std::pair<std::string, float>> result(db_.size()); {
			size_t i = 0;
			for (auto &line : db_) {
				result[i].first = line.first;
				Compare(feat, line.second, &result[i].second);
				i++;
			}
		}

		std::partial_sort(result.begin(), result.begin() + 1, result.end(), [](
			const std::pair<std::string, float> &a, const std::pair<std::string, float> &b) -> bool {
			return a.second > b.second;
		});

		query_result->name_ = result[0].first;
		query_result->sim_  = result[0].second;
		
		return 0;
	}


private:
	std::map<int64_t, std::vector<float>> features_db_;
	std::map<int64_t, std::string> names_db_;
	std::map<std::string, std::vector<float>> db_;
	int64_t max_index_ = 0;
};

FaceDatabase::FaceDatabase() {
	impl_ = new FaceDatabase::Impl();
}

FaceDatabase::~FaceDatabase() {
	if (impl_) {
		delete impl_;
		impl_ = nullptr;
	}
}

bool FaceDatabase::Save(const char* path) const {
	std::cout << "start save data." << std::endl;
	std::string db_name = std::string(path) + "/db";
	FileWriter ofile(db_name.c_str(), FileWriter::Binary);
	if (!ofile.is_opened()){
		std::cout << "open database failed." << std::endl;
		return false;
	} 
	return impl_->Save(ofile);
}

bool FaceDatabase::Load(const char* path) {
	std::string db_name = std::string(path) + "/db";
	FileReader ifile(db_name.c_str(), FileWriter::Binary);
	if (!ifile.is_opened()) return false;
	return impl_->Load(ifile);
}

int64_t FaceDatabase::Insert(const std::vector<float>& feat, const std::string& name) {
	return impl_->Insert(name, feat);
}

int FaceDatabase::Delete(const std::string& name) {
	return impl_->Delete(name);
}

int64_t FaceDatabase::QueryTop(const std::vector<float>& feat,
	QueryResult* query_result) {
	return impl_->QueryTop(feat, query_result);
}

void FaceDatabase::Clear() {
	impl_->Clear();
}



}

