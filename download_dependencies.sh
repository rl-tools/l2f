git submodule update --init external/rl-tools
(cd external/rl-tools/static/ui_server/generic && ./download_dependencies.sh)
(cd external/json/nlohmann && wget https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp)