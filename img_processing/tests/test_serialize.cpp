#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <bitset>
#include <string>
#include "alert.h"
#include "object_type_enum.h"
#include "manager.h"
#include "alerter.h"

using namespace std;

string toBinaryString(vector<uint8_t> vec)
{
    std::string binaryString;
    for (uint8_t val : vec) {
        // Convert uint8_t to binary representation
        std::bitset<8> bits(val);
        // Append binary representation to the string
        binaryString += bits.to_string();
    }
    return binaryString;
}

TEST(AlertTest, checkBinaryValue)
{
    Alert alert(false, 1, 0, 2.5, 3.0);
    string binary =
        "0001000000000000000000000010000001000000000000000000000001000000010000"
        "00";
    vector<uint8_t> v = alert.serialize();
    EXPECT_EQ(binary, toBinaryString(v));
}