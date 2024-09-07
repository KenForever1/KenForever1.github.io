### 序列化Map, 与nlohmann json输出相同

```c++
#include <map>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

using namespace rapidjson;
using rapidjson::StringBuffer;
using rapidjson::Writer;
using namespace rapidjson;

void serializeMaps(const std::map<int, std::string> &m, rapidjson::Writer <rapidjson::StringBuffer> *writer) {
    writer->StartArray();
    for (std::map<int, std::string>::const_iterator it = m.begin(); it != m.end(); ++it){
        writer->StartArray();
        writer->Int(it->first);
        writer->String(it->second.c_str());
        writer->EndArray();
    }
    writer->EndArray();
}

int main() {
    std::map<int, std::string> m;
    m.emplace(1, "111");
    m.emplace(2, "222");

    StringBuffer sb;
    Writer<StringBuffer> writer(sb);

    writer.StartObject();
    writer.Key("m");
    serializeMaps(m, &writer);
    writer.EndObject();

    puts(sb.GetString());
}
```