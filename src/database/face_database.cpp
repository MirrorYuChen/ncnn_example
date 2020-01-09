#include "face_database.h"
#include <iostream>

class FaceDatabase::Impl {
public:
	Impl() {
		max_index_ = 0;
		features_db_.clear();
	}

	~Impl() {

	}
	
	bool Save(StreamWriter &features_writer, StreamWriter &names_writer) const {
		const uint64_t num_feat = features_db_.size();
		const uint64_t num_name = names_db_.size();
		const uint64_t dim_feat = kFaceFeatureDim;
		const uint64_t dim_name = kFaceNameDim;

		Write(features_writer, num_feat);
		Write(features_writer, dim_feat);
		Write(names_writer, num_name);
		Write(names_writer, dim_name);

		for (auto &line : features_db_) {
			auto &index_feat = line.first;
			auto &feature = line.second;
			// do save
			Write(features_writer, index_feat);
			Write(features_writer, &feature[0], size_t(dim_feat));
		}

		for (auto &line : names_db_) {
			auto &index_name = line.first;
			auto &name = line.second;

			char name_arr[kFaceNameDim];
			sprintf(name_arr, "%s", name.c_str());
			// do save
			Write(names_writer, index_name);
			Write(names_writer, name_arr, size_t(dim_name));
			std::cout << "write name: " << name << std::endl;
		}

		std::cout << "FaceDatabase Loaded " << num_feat << " faces" << std::endl;
		return true;
	}

	bool Load(StreamReader &features_reader, StreamReader &names_reader) {
		uint64_t num_feat = 0;
		uint64_t num_name = 0;
		uint64_t dim_feat = 0;
		uint64_t dim_name = 0;
		Read(features_reader, num_feat);
		Read(features_reader, dim_feat);
		Read(names_reader, num_name);
		Read(names_reader, dim_name);

		features_db_.clear();
		names_db_.clear();
		max_index_ = -1;

		for (size_t i = 0; i < num_feat; ++i) {
			int64_t index_feat = 0;
			int64_t index_name = 0;
			std::vector<float> feat(kFaceFeatureDim);
			char name_arr[kFaceNameDim];

			Read(features_reader, index_feat);
			Read(features_reader, &feat[0], size_t(dim_feat));

			Read(names_reader, index_name);
			Read(names_reader, name_arr, size_t(dim_name));
			std::cout << "name is: " << name_arr << std::endl;

			features_db_.insert(std::make_pair(index_feat, feat));
			names_db_.insert(std::make_pair(index_name, std::string(name_arr)));
			max_index_ = (max_index_ > index_feat ? max_index_ : index_feat);
		}
		++max_index_;

		std::cout << "FaceDatabase Loaded " << num_feat << " faces" << std::endl;

		return true;
	}

	int64_t Insert(const std::vector<float> &feat, const std::string& name) {
		int64_t new_index = max_index_++;
		std::cout << "new index is: " << new_index << std::endl;
		features_db_.insert(std::make_pair(new_index, feat));
		names_db_.insert(std::make_pair(new_index, name));
		return new_index;
	}

	int Delete(int64_t index) {
		return int(features_db_.erase(index) && names_db_.erase(index));
	}

	void Clear() {
		features_db_.clear();
		names_db_.clear();
		max_index_ = 0;
	}

	size_t Count() const {
		return features_db_.size();
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
		std::vector<std::pair<int64_t, float>> result(features_db_.size()); {
			size_t i = 0;
			for (auto &line : features_db_) {
				result[i].first = line.first;
				Compare(feat, line.second, &result[i].second);
				i++;
			}
		}

		std::partial_sort(result.begin(), result.begin() + 1, result.end(), [](
			const std::pair<int64_t, float> &a, const std::pair<int64_t, float> &b) -> bool {
			return a.second > b.second;
		});

		std::map<int64_t, std::string>::iterator it = names_db_.find(result[0].first);
		if (it != names_db_.end()) {
			query_result->name_ = it->second;
		}

		query_result->sim_ = result[0].second;
		
		return 0;
	}


private:
	std::map<int64_t, std::vector<float>> features_db_;
	std::map<int64_t, std::string> names_db_;
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
	std::string features_db_name = std::string(path) + "/features_db.db";
	std::string names_db_name = std::string(path) + "/names_db.db";
	FileWriter ofile_feature(features_db_name.c_str(), FileWriter::Binary);
	FileWriter ofile_name(names_db_name.c_str(), FileWriter::Binary);
	if (!ofile_feature.is_opened() || !ofile_name.is_opened()){
		std::cout << "open database failed." << std::endl;
		return false;
	} 
	return impl_->Save(ofile_feature, ofile_name);
}

bool FaceDatabase::Load(const char* path) {
	std::string features_db_name = std::string(path) + "/features_db.db";
	std::string names_db_name = std::string(path) + "/names_db.db";
	FileReader ifile_feature(features_db_name.c_str(), FileWriter::Binary);
	FileReader ifile_name(names_db_name.c_str(), FileWriter::Binary);
	if (!ifile_feature.is_opened() || !ifile_name.is_opened()) return false;
	return impl_->Load(ifile_feature, ifile_name);
}

int64_t FaceDatabase::Insert(const std::vector<float>& feat, const std::string& name) {
	return impl_->Insert(feat, name);
}

int FaceDatabase::Delete(int64_t index) {
	return impl_->Delete(index);
}

int64_t FaceDatabase::QueryTop(const std::vector<float>& feat,
	QueryResult* query_result) {
	return impl_->QueryTop(feat, query_result);
}

void FaceDatabase::Clear() {
	impl_->Clear();
}

size_t FaceDatabase::Count() const {
	return impl_->Count();
}

