#include "io/labels.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

using yolox::io::DefaultCocoLabels;
using yolox::io::LabelFor;
using yolox::io::LoadLabels;

namespace {

// テスト用の一時ラベルファイルを作り、スコープを抜けたら削除するRAIIヘルパー。
class TempLabelsFile {
public:
    explicit TempLabelsFile(const std::string& content) {
        path_ = (std::filesystem::temp_directory_path() /
                 ("yolox_test_labels_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
                  "_" + std::to_string(counter_++) + ".txt"))
                    .string();
        std::ofstream file(path_);
        file << content;
    }
    ~TempLabelsFile() { std::remove(path_.c_str()); }

    const std::string& path() const { return path_; }

private:
    std::string path_;
    static inline int counter_ = 0;
};

}  // namespace

TEST(Labels, DefaultCocoLabelsHas80Entries) {
    const auto labels = DefaultCocoLabels();
    ASSERT_EQ(labels.size(), 80U);
    EXPECT_EQ(labels.front(), "person");
    EXPECT_EQ(labels.back(), "toothbrush");
}

TEST(Labels, LoadLabelsSkipsBlankAndCommentLinesAndTrims) {
    TempLabelsFile file(
        "  cat  \n"
        "\n"
        "# comment\n"
        "dog\n"
        "   \n");
    const auto labels = LoadLabels(file.path());

    ASSERT_EQ(labels.size(), 2U);
    EXPECT_EQ(labels[0], "cat");
    EXPECT_EQ(labels[1], "dog");
}

TEST(Labels, LoadLabelsThrowsOnMissingFile) {
    EXPECT_THROW(LoadLabels("this/path/does/not/exist.txt"), std::runtime_error);
}

TEST(Labels, LoadLabelsThrowsOnEmptyFile) {
    TempLabelsFile file("# only a comment\n\n");
    EXPECT_THROW(LoadLabels(file.path()), std::runtime_error);
}

TEST(Labels, LabelForReturnsNameWithinRange) {
    const std::vector<std::string> labels = {"cat", "dog"};
    EXPECT_EQ(LabelFor(labels, 0), "cat");
    EXPECT_EQ(LabelFor(labels, 1), "dog");
}

TEST(Labels, LabelForFallsBackForOutOfRange) {
    const std::vector<std::string> labels = {"cat", "dog"};
    EXPECT_EQ(LabelFor(labels, 99), "class_99");
    EXPECT_EQ(LabelFor(labels, -1), "class_-1");
}
