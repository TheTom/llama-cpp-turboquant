#include "server-tools.h"

#include "download.h"
#include "http.h"

#include <sheredom/subprocess.h>

#include <cctype>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <thread>
#include <chrono>
#include <atomic>
#include <cstring>
#include <climits>
#include <algorithm>
#include <cstdlib>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#endif

#ifdef _WIN32
extern char ** _environ;
#else
extern char ** environ;
#endif

namespace fs = std::filesystem;

//
// internal helpers
//

static std::vector<char *> to_cstr_vec(const std::vector<std::string> & v) {
    std::vector<char *> r;
    r.reserve(v.size() + 1);
    for (const auto & s : v) {
        r.push_back(const_cast<char *>(s.c_str()));
    }
    r.push_back(nullptr);
    return r;
}

struct run_proc_result {
    std::string output;
    int  exit_code = -1;
    bool timed_out = false;
};

static run_proc_result run_process(
        const std::vector<std::string> & args,
        size_t max_output,
        int timeout_secs,
        const std::vector<std::string> * environment = nullptr) {
    run_proc_result res;

    subprocess_s proc;
    auto argv = to_cstr_vec(args);

    int options = subprocess_option_no_window
                | subprocess_option_combined_stdout_stderr
                | subprocess_option_search_user_path;
    if (environment == nullptr) {
        options |= subprocess_option_inherit_environment;
    }

    std::vector<char *> envp;
    if (environment != nullptr) {
        envp = to_cstr_vec(*environment);
    }
    const int create_result = environment == nullptr
        ? subprocess_create(argv.data(), options, &proc)
        : subprocess_create_ex(argv.data(), options, envp.data(), &proc);
    if (create_result != 0) {
        res.output = "failed to spawn process";
        return res;
    }

    std::atomic<bool> done{false};
    std::atomic<bool> timed_out{false};

    std::thread timeout_thread([&]() {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_secs);
        while (!done.load()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                timed_out.store(true);
                subprocess_terminate(&proc);
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    FILE * f = subprocess_stdout(&proc);
    std::string output;
    bool truncated = false;
    if (f) {
        char buf[4096];
        while (fgets(buf, sizeof(buf), f) != nullptr) {
            if (!truncated) {
                size_t len = strlen(buf);
                if (output.size() + len <= max_output) {
                    output.append(buf, len);
                } else {
                    output.append(buf, max_output - output.size());
                    truncated = true;
                }
            }
        }
    }

    done.store(true);
    if (timeout_thread.joinable()) {
        timeout_thread.join();
    }

    subprocess_join(&proc, &res.exit_code);
    subprocess_destroy(&proc);

    res.output    = output;
    res.timed_out = timed_out.load();
    if (truncated) {
        res.output += "\n[output truncated]";
    }
    return res;
}

static std::vector<std::string> sanitized_git_environment() {
    std::vector<std::string> result;
#ifdef _WIN32
    char ** current = _environ;
#else
    char ** current = environ;
#endif
    for (; current != nullptr && *current != nullptr; ++current) {
        const std::string entry(*current);
        const size_t separator = entry.find('=');
        std::string key = entry.substr(0, separator);
#ifdef _WIN32
        std::transform(key.begin(), key.end(), key.begin(),
            [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
#endif
        if (key.rfind("GIT_", 0) != 0) {
            result.push_back(entry);
        }
    }
    result.push_back("GIT_OPTIONAL_LOCKS=0");
    result.push_back("GIT_TERMINAL_PROMPT=0");
#ifdef _WIN32
    std::sort(result.begin(), result.end());
#endif
    return result;
}

static std::string url_encode(const std::string & value) {
    static constexpr char hex[] = "0123456789ABCDEF";
    std::string result;
    result.reserve(value.size());
    for (unsigned char c : value) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            result.push_back(static_cast<char>(c));
        } else {
            result.push_back('%');
            result.push_back(hex[c >> 4]);
            result.push_back(hex[c & 0x0f]);
        }
    }
    return result;
}

static bool is_blocked_ipv4(uint32_t address) {
    const uint32_t ip = ntohl(address);
    return (ip >> 24) == 0 ||
           (ip >> 24) == 10 ||
           (ip >> 22) == 0x0191 ||       // 100.64.0.0/10
           (ip >> 24) == 127 ||
           (ip >> 16) == 0xa9fe ||       // 169.254.0.0/16
           (ip >> 20) == 0x0ac1 ||       // 172.16.0.0/12
           (ip >>  8) == 0xc00000 ||      // 192.0.0.0/24
           (ip >>  8) == 0xc00002 ||      // 192.0.2.0/24
           (ip >> 16) == 0xc0a8 ||       // 192.168.0.0/16
           (ip & 0xfffe0000) == 0xc6120000 || // 198.18.0.0/15
           (ip >>  8) == 0xc63364 ||      // 198.51.100.0/24
           (ip >>  8) == 0xcb0071 ||      // 203.0.113.0/24
           (ip >> 28) >= 0x0e;           // multicast and reserved
}

static bool is_blocked_address(const sockaddr * address) {
    if (address->sa_family == AF_INET) {
        const auto * ipv4 = reinterpret_cast<const sockaddr_in *>(address);
        return is_blocked_ipv4(ipv4->sin_addr.s_addr);
    }
    if (address->sa_family == AF_INET6) {
        const auto * ipv6 = reinterpret_cast<const sockaddr_in6 *>(address);
        const auto & bytes = ipv6->sin6_addr.s6_addr;
        if (IN6_IS_ADDR_UNSPECIFIED(&ipv6->sin6_addr) ||
            IN6_IS_ADDR_LOOPBACK(&ipv6->sin6_addr) ||
            IN6_IS_ADDR_LINKLOCAL(&ipv6->sin6_addr) ||
            IN6_IS_ADDR_MULTICAST(&ipv6->sin6_addr) ||
            (bytes[0] & 0xfe) == 0xfc) {
            return true;
        }
        if (IN6_IS_ADDR_V4MAPPED(&ipv6->sin6_addr)) {
            uint32_t mapped;
            std::memcpy(&mapped, bytes + 12, sizeof(mapped));
            return is_blocked_ipv4(mapped);
        }
    }
    return false;
}

static std::string normalize_http_url(std::string url) {
    const size_t fragment = url.find('#');
    if (fragment != std::string::npos) {
        url.resize(fragment);
    }
    const size_t scheme = url.find("://");
    if (scheme == std::string::npos) {
        throw std::invalid_argument("URL must include http:// or https://");
    }
    const size_t authority = scheme + 3;
    const size_t slash = url.find('/', authority);
    const size_t query = url.find('?', authority);
    const size_t authority_end = std::min(
        slash == std::string::npos ? url.size() : slash,
        query == std::string::npos ? url.size() : query);
    if (url.substr(authority, authority_end - authority).find('@') != std::string::npos) {
        throw std::invalid_argument("URL credentials are not allowed");
    }
    if (slash == std::string::npos || (query != std::string::npos && query < slash)) {
        url.insert(authority_end, "/");
    }
    return url;
}

static void validate_public_url(const std::string & url) {
    const auto normalized = normalize_http_url(url);
    const auto parts = common_http_parse_url(normalized);
    if (parts.scheme != "http" && parts.scheme != "https") {
        throw std::invalid_argument("only HTTP and HTTPS URLs are supported");
    }

    addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo * addresses = nullptr;
    const int rc = getaddrinfo(parts.host.c_str(), nullptr, &hints, &addresses);
    if (rc != 0 || addresses == nullptr) {
        throw std::runtime_error("cannot resolve URL host");
    }

    bool blocked = false;
    for (addrinfo * current = addresses; current != nullptr; current = current->ai_next) {
        if (is_blocked_address(current->ai_addr)) {
            blocked = true;
            break;
        }
    }
    freeaddrinfo(addresses);
    if (blocked) {
        throw std::invalid_argument("URL resolves to a private or reserved address");
    }
}

static std::string resolve_redirect_url(const std::string & current, const std::string & location) {
    if (location.rfind("http://", 0) == 0 || location.rfind("https://", 0) == 0) {
        return location;
    }
    const auto parts = common_http_parse_url(current);
    const std::string origin = parts.scheme + "://" + parts.host +
        ((parts.scheme == "http" && parts.port == 80) || (parts.scheme == "https" && parts.port == 443)
            ? "" : ":" + std::to_string(parts.port));
    if (!location.empty() && location[0] == '/') {
        return origin + location;
    }
    const size_t last_slash = parts.path.rfind('/');
    return origin + (last_slash == std::string::npos ? "/" : parts.path.substr(0, last_slash + 1)) + location;
}

struct http_fetch_result {
    std::string url;
    std::string body;
    std::string content_type;
    int status = 0;
    bool truncated = false;
};

static http_fetch_result fetch_public_url(std::string url, size_t max_size, int timeout_secs) {
    static constexpr int max_redirects = 5;

    for (int redirect = 0; redirect <= max_redirects; ++redirect) {
        url = normalize_http_url(url);
        validate_public_url(url);
        auto [client, parts] = common_http_client(url);
        client.set_follow_location(false);
        client.set_connection_timeout(timeout_secs, 0);
        client.set_read_timeout(timeout_secs, 0);
        client.set_write_timeout(timeout_secs, 0);

        http_fetch_result result;
        result.url = url;
        httplib::Headers headers = {
            {"Accept", "text/html, text/plain, application/json, application/xml;q=0.9, */*;q=0.1"},
        };
        auto response = client.Get(
            parts.path,
            headers,
            [&](const httplib::Response & res) {
                result.status = res.status;
                result.content_type = res.get_header_value("Content-Type");
                return true;
            },
            [&](const char * data, size_t length) {
                const size_t remaining = max_size - result.body.size();
                if (length > remaining) {
                    result.body.append(data, remaining);
                    result.truncated = true;
                    return false;
                }
                result.body.append(data, length);
                return true;
            });

        if (!response && !result.truncated) {
            throw std::runtime_error("HTTP request failed");
        }
        if (result.status >= 300 && result.status < 400 && response && response->has_header("Location")) {
            if (redirect == max_redirects) {
                throw std::runtime_error("too many HTTP redirects");
            }
            url = resolve_redirect_url(url, response->get_header_value("Location"));
            continue;
        }
        if (result.status < 200 || result.status >= 300) {
            throw std::runtime_error("HTTP request returned status " + std::to_string(result.status));
        }

        std::string content_type = result.content_type;
        std::transform(content_type.begin(), content_type.end(), content_type.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (!content_type.empty() &&
            content_type.rfind("text/", 0) != 0 &&
            content_type.find("json") == std::string::npos &&
            content_type.find("xml") == std::string::npos &&
            content_type.find("javascript") == std::string::npos) {
            throw std::runtime_error("unsupported content type: " + result.content_type);
        }
        return result;
    }
    throw std::runtime_error("too many HTTP redirects");
}

json server_tool::to_json() {
    return {
        {"display_name", display_name},
        {"tool", name},
        {"type", "builtin"},
        {"permissions", json{
            {"write", permission_write}
        }},
        {"definition", get_definition()},
    };
}

//
// read_file: read a file with optional line range and line-number prefix
//

static constexpr size_t SERVER_TOOL_READ_FILE_MAX_SIZE = 16 * 1024; // 16 KB

struct server_tool_read_file : server_tool {
    server_tool_read_file() {
        name = "read_file";
        display_name = "Read file";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Read the contents of a file. Optionally specify a 1-based line range. "
                                "If append_loc is true, each line is prefixed with its line number (e.g. \"1\u2192 ...\")."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",       {{"type", "string"},  {"description", "Path to the file"}}},
                        {"start_line", {{"type", "integer"}, {"description", "First line to read, 1-based (default: 1)"}}},
                        {"end_line",   {{"type", "integer"}, {"description", "Last line to read, 1-based inclusive (default: end of file)"}}},
                        {"append_loc", {{"type", "boolean"}, {"description", "Prefix each line with its line number"}}},
                    }},
                    {"required", json::array({"path"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path  = params.at("path").get<std::string>();
        int  start_line   = json_value(params, "start_line", 1);
        int  end_line     = json_value(params, "end_line",  -1); // -1 = no limit
        bool append_loc   = json_value(params, "append_loc", false);

        std::error_code ec;
        uintmax_t file_size = fs::file_size(path, ec);
        if (ec) {
            return {{"error", "cannot stat file: " + ec.message()}};
        }
        if (file_size > SERVER_TOOL_READ_FILE_MAX_SIZE && end_line == -1) {
            return {{"error", string_format(
                "file too large (%zu bytes, max %zu). Use start_line/end_line to read a portion.",
                (size_t)file_size, SERVER_TOOL_READ_FILE_MAX_SIZE)}};
        }

        std::ifstream f(path);
        if (!f) {
            return {{"error", "failed to open file: " + path}};
        }

        std::string result;
        std::string line;
        int lineno = 0;

        while (std::getline(f, line)) {
            lineno++;
            if (lineno < start_line) continue;
            if (end_line != -1 && lineno > end_line) break;

            std::string out_line;
            if (append_loc) {
                out_line = std::to_string(lineno) + "\u2192 " + line + "\n";
            } else {
                out_line = line + "\n";
            }

            if (result.size() + out_line.size() > SERVER_TOOL_READ_FILE_MAX_SIZE) {
                result += "[output truncated]";
                break;
            }
            result += out_line;
        }

        return {{"plain_text_response", result}};
    }
};

//
// file_glob_search: find files matching a glob pattern under a base directory
//

static constexpr size_t SERVER_TOOL_FILE_SEARCH_MAX_RESULTS = 100;

struct server_tool_file_glob_search : server_tool {
    server_tool_file_glob_search() {
        name = "file_glob_search";
        display_name = "File search";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Recursively search for files matching a glob pattern under a directory."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",    {{"type", "string"}, {"description", "Base directory to search in"}}},
                        {"include", {{"type", "string"}, {"description", "Glob pattern for files to include (e.g. \"**/*.cpp\"). Default: **"}}},
                        {"exclude", {{"type", "string"}, {"description", "Glob pattern for files to exclude"}}},
                    }},
                    {"required", json::array({"path"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string base    = params.at("path").get<std::string>();
        std::string include = json_value(params, "include", std::string("**"));
        std::string exclude = json_value(params, "exclude", std::string(""));

        std::ostringstream output_text;
        size_t count = 0;

        std::error_code ec;
        for (const auto & entry : fs::recursive_directory_iterator(base,
                fs::directory_options::skip_permission_denied, ec)) {
            if (!entry.is_regular_file()) continue;

            std::string rel = fs::relative(entry.path(), base, ec).string();
            if (ec) continue;
            std::replace(rel.begin(), rel.end(), '\\', '/');

            if (!glob_match(include, rel)) continue;
            if (!exclude.empty() && glob_match(exclude, rel)) continue;

            output_text << entry.path().string() << "\n";
            if (++count >= SERVER_TOOL_FILE_SEARCH_MAX_RESULTS) {
                break;
            }
        }

        output_text << "\n---\nTotal matches: " << count << "\n";

        return {{"plain_text_response", output_text.str()}};
    }
};

//
// grep_search: search for a regex pattern in files
//

static constexpr size_t SERVER_TOOL_GREP_SEARCH_MAX_RESULTS = 100;

struct server_tool_grep_search : server_tool {
    server_tool_grep_search() {
        name = "grep_search";
        display_name = "Grep search";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Search for a regex pattern in files under a path. Returns matching lines."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",                {{"type", "string"},  {"description", "File or directory to search in"}}},
                        {"pattern",             {{"type", "string"},  {"description", "Regular expression pattern to search for"}}},
                        {"include",             {{"type", "string"},  {"description", "Glob pattern to filter files (default: **)"}}},
                        {"exclude",             {{"type", "string"},  {"description", "Glob pattern to exclude files"}}},
                        {"return_line_numbers", {{"type", "boolean"}, {"description", "If true, include line numbers in results"}}},
                    }},
                    {"required", json::array({"path", "pattern"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path    = params.at("path").get<std::string>();
        std::string pat_str = params.at("pattern").get<std::string>();
        std::string include = json_value(params, "include", std::string("**"));
        std::string exclude = json_value(params, "exclude", std::string(""));
        bool show_lineno    = json_value(params, "return_line_numbers", false);

        std::regex pattern;
        try {
            pattern = std::regex(pat_str);
        } catch (const std::regex_error & e) {
            return {{"error", std::string("invalid regex: ") + e.what()}};
        }

        std::ostringstream output_text;
        size_t total = 0;

        auto search_file = [&](const fs::path & fpath) {
            std::ifstream f(fpath);
            if (!f) return;
            std::string line;
            int lineno = 0;
            while (std::getline(f, line) && total < SERVER_TOOL_GREP_SEARCH_MAX_RESULTS) {
                lineno++;
                if (std::regex_search(line, pattern)) {
                    output_text << fpath.string() << ":";
                    if (show_lineno) {
                        output_text << lineno << ":";
                    }
                    output_text << line << "\n";
                    total++;
                }
            }
        };

        std::error_code ec;
        if (fs::is_regular_file(path, ec)) {
            search_file(path);
        } else if (fs::is_directory(path, ec)) {
            for (const auto & entry : fs::recursive_directory_iterator(path,
                    fs::directory_options::skip_permission_denied, ec)) {
                if (!entry.is_regular_file()) continue;
                if (total >= SERVER_TOOL_GREP_SEARCH_MAX_RESULTS) break;

                std::string rel = fs::relative(entry.path(), path, ec).string();
                if (ec) continue;
                std::replace(rel.begin(), rel.end(), '\\', '/');

                if (!glob_match(include, rel)) continue;
                if (!exclude.empty() && glob_match(exclude, rel)) continue;

                search_file(entry.path());
            }
        } else {
            return {{"error", "path does not exist: " + path}};
        }

        output_text << "\n\n---\nTotal matches: " << total << "\n";

        return {{"plain_text_response", output_text.str()}};
    }
};

//
// web_search: search the web through a SearXNG instance
//

static constexpr size_t SERVER_TOOL_WEB_SEARCH_MAX_RESPONSE_SIZE = 256 * 1024;
static constexpr int    SERVER_TOOL_WEB_SEARCH_MAX_RESULTS       = 20;

struct server_tool_web_search : server_tool {
    server_tool_web_search() {
        name = "web_search";
        display_name = "Web search";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description",
                    "Search the public web for current or external information via SearXNG. "
                    "Returns titles, URLs, and snippets. After choosing a useful URL, call fetch_url to read the page. "
                    "Do not use for local files or Git history."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"query",      {{"type", "string"},  {"description", "Search query"}}},
                        {"count",      {{"type", "integer"}, {"minimum", 1}, {"maximum", 20}, {"description", "Maximum number of results (default 10, max 20)"}}},
                        {"categories", {{"type", "string"},  {"description", "Comma-separated SearXNG categories (default: general)"}}},
                        {"language",   {{"type", "string"},  {"description", "Search language (default: all)"}}},
                        {"safesearch", {{"type", "integer"}, {"enum", json::array({0, 1, 2})}, {"description", "Safe search level: 0, 1, or 2 (default: 1)"}}},
                    }},
                    {"required", json::array({"query"})},
                    {"additionalProperties", false},
                }},
            }},
        };
    }

    json invoke(json params) override {
        const std::string query = params.at("query").get<std::string>();
        if (query.empty()) {
            return {{"error", "query must not be empty"}};
        }

        const int count = std::clamp(
            json_value(params, "count", 10), 1, SERVER_TOOL_WEB_SEARCH_MAX_RESULTS);
        const int safesearch = std::clamp(json_value(params, "safesearch", 1), 0, 2);
        const std::string categories = json_value(params, "categories", std::string("general"));
        const std::string language = json_value(params, "language", std::string("all"));
        const char * configured_url = std::getenv("SEARXNG_URL");
        std::string base_url = configured_url && configured_url[0]
            ? configured_url : "http://localhost:9090";
        while (!base_url.empty() && base_url.back() == '/') {
            base_url.pop_back();
        }
        try {
            const auto parts = common_http_parse_url(normalize_http_url(base_url));
            if (parts.scheme != "http" && parts.scheme != "https") {
                return {{"error", "SEARXNG_URL must use HTTP or HTTPS"}};
            }
        } catch (const std::exception & e) {
            return {{"error", std::string("invalid SEARXNG_URL: ") + e.what()}};
        }

        const std::string url = base_url + "/search?format=json&q=" + url_encode(query) +
            "&categories=" + url_encode(categories) +
            "&language=" + url_encode(language) +
            "&safesearch=" + std::to_string(safesearch);

        try {
            common_remote_params remote_params;
            remote_params.timeout = 10;
            remote_params.max_size = SERVER_TOOL_WEB_SEARCH_MAX_RESPONSE_SIZE;
            remote_params.headers.push_back({"Accept", "application/json"});
            const auto [status, body] = common_remote_get_content(url, remote_params);
            if (status < 200 || status >= 300) {
                return {{"error", "SearXNG returned HTTP status " + std::to_string(status)}};
            }

            const json response = json::parse(body.begin(), body.end());
            if (!response.contains("results") || !response["results"].is_array()) {
                return {{"error", "SearXNG response does not contain a results array"}};
            }

            std::ostringstream output;
            int emitted = 0;
            for (const auto & item : response["results"]) {
                if (emitted >= count) {
                    break;
                }
                const std::string item_url = item.value("url", "");
                if (item_url.empty()) {
                    continue;
                }
                output << emitted + 1 << ". " << item.value("title", item_url) << "\n"
                       << "URL: " << item_url << "\n";
                const std::string content = item.value("content", "");
                if (!content.empty()) {
                    output << content << "\n";
                }
                output << "\n";
                emitted++;
            }
            if (emitted == 0) {
                output << "No results found.\n";
            }
            return {{"plain_text_response", output.str()}};
        } catch (const std::exception & e) {
            return {{"error", std::string("web search failed: ") + e.what()}};
        }
    }
};

//
// fetch_url: fetch bounded public HTTP(S) text content
//

static constexpr size_t SERVER_TOOL_FETCH_URL_DEFAULT_MAX_SIZE = 64 * 1024;
static constexpr size_t SERVER_TOOL_FETCH_URL_MAX_SIZE         = 256 * 1024;

struct server_tool_fetch_url : server_tool {
    server_tool_fetch_url() {
        name = "fetch_url";
        display_name = "Fetch URL";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description",
                    "Read textual content from a known public HTTP or HTTPS URL. "
                    "This does not search the web; use web_search when the URL is unknown. "
                    "Private, loopback, and link-local addresses are blocked."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"url",             {{"type", "string"},  {"description", "Public HTTP or HTTPS URL"}}},
                        {"max_output_size", {{"type", "integer"}, {"minimum", 1}, {"maximum", 262144}, {"description", "Maximum response size in bytes (default 65536, max 262144)"}}},
                        {"timeout",         {{"type", "integer"}, {"minimum", 1}, {"maximum", 30}, {"description", "Timeout in seconds (default 10, max 30)"}}},
                    }},
                    {"required", json::array({"url"})},
                    {"additionalProperties", false},
                }},
            }},
        };
    }

    json invoke(json params) override {
        const std::string url = params.at("url").get<std::string>();
        const size_t max_size = static_cast<size_t>(std::clamp(
            json_value(params, "max_output_size", static_cast<int>(SERVER_TOOL_FETCH_URL_DEFAULT_MAX_SIZE)),
            1, static_cast<int>(SERVER_TOOL_FETCH_URL_MAX_SIZE)));
        const int timeout = std::clamp(json_value(params, "timeout", 10), 1, 30);

        try {
            const auto result = fetch_public_url(url, max_size, timeout);
            std::ostringstream output;
            output << "URL: " << result.url << "\n"
                   << "Content-Type: " << (result.content_type.empty() ? "unknown" : result.content_type) << "\n\n"
                   << result.body;
            if (result.truncated) {
                output << "\n[output truncated]";
            }
            return {{"plain_text_response", output.str()}};
        } catch (const std::exception & e) {
            return {{"error", std::string("fetch failed: ") + e.what()}};
        }
    }
};

//
// exec_shell_command: run an arbitrary shell command
//

static constexpr size_t SERVER_TOOL_EXEC_SHELL_COMMAND_DEFAULT_OUTPUT_SIZE = 64 * 1024;
static constexpr size_t SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE     = 256 * 1024;
static constexpr int    SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT         = 60;

static std::vector<std::string> build_shell_args(const std::string & shell, const std::string & command) {
#ifdef _WIN32
    if (shell == "powershell" || shell == "pwsh") {
        return {"powershell", "-NoProfile", "-NonInteractive", "-Command", command};
    }
    if (shell == "bash") {
        return {"bash", "-lc", command};
    }
    return {"cmd", "/c", command};
#else
    if (shell == "powershell" || shell == "pwsh") {
        return {"pwsh", "-NoProfile", "-NonInteractive", "-Command", command};
    }
    if (shell == "cmd") {
        return {"cmd.exe", "/c", command};
    }
    if (shell == "bash") {
        return {"bash", "-lc", command};
    }
    return {"sh", "-c", command};
#endif
}

struct server_tool_exec_shell_command : server_tool {
    server_tool_exec_shell_command() {
        name = "exec_shell_command";
        display_name = "Execute shell command";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description",
                    "Run a terminal command and return combined stdout/stderr. "
                    "Prefer dedicated tools for file edits and Git inspection when available. "
                    "Use shell=bash for Bash, shell=powershell for PowerShell, or omit shell for the platform default."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"command", {{"type", "string"}, {"description", "Shell command to execute"}}},
                        {"shell", {
                            {"type", "string"},
                            {"enum", json::array({"auto", "bash", "powershell", "cmd", "sh"})},
                            {"description", "Shell to use (default: auto)"},
                        }},
                        {"timeout", {
                            {"type", "integer"},
                            {"minimum", 1},
                            {"maximum", SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT},
                            {"description", string_format("Timeout in seconds (default 10, max %d)", SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT)},
                        }},
                        {"max_output_size", {
                            {"type", "integer"},
                            {"minimum", 1},
                            {"maximum", (int) SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE},
                            {"description", string_format("Maximum output size in bytes (default %zu)", SERVER_TOOL_EXEC_SHELL_COMMAND_DEFAULT_OUTPUT_SIZE)},
                        }},
                    }},
                    {"required", json::array({"command"})},
                    {"additionalProperties", false},
                }},
            }},
        };
    }

    json invoke(json params) override {
        const std::string command = params.at("command").get<std::string>();
        std::string shell = json_value(params, "shell", std::string("auto"));
        int timeout = std::clamp(json_value(params, "timeout", 10), 1, SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT);
        size_t max_output = static_cast<size_t>(std::clamp(
            json_value(params, "max_output_size", (int) SERVER_TOOL_EXEC_SHELL_COMMAND_DEFAULT_OUTPUT_SIZE),
            1, (int) SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE));

        if (shell == "auto") {
#ifdef _WIN32
            shell = "cmd";
#else
            shell = "sh";
#endif
        }

        auto res = run_process(build_shell_args(shell, command), max_output, timeout);

        std::string text_output = res.output;
        text_output += string_format("\n[exit code: %d]", res.exit_code);
        if (res.timed_out) {
            text_output += " [exit due to timed out]";
        }

        return {{"plain_text_response", text_output}};
    }
};

//
// run_python: execute a Python snippet or script file
//

static constexpr size_t SERVER_TOOL_RUN_PYTHON_DEFAULT_OUTPUT_SIZE = 64 * 1024;
static constexpr size_t SERVER_TOOL_RUN_PYTHON_MAX_OUTPUT_SIZE     = 256 * 1024;
static constexpr int    SERVER_TOOL_RUN_PYTHON_MAX_TIMEOUT         = 60;

struct server_tool_run_python : server_tool {
    server_tool_run_python() {
        name = "run_python";
        display_name = "Run Python";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description",
                    "Execute Python code or a Python script file and return combined stdout/stderr. "
                    "Provide either code or path. Prefer this over wrapping python in exec_shell_command."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"code", {
                            {"type", "string"},
                            {"description", "Python source to execute (mutually exclusive with path)"},
                        }},
                        {"path", {
                            {"type", "string"},
                            {"description", "Path to a .py file to execute (mutually exclusive with code)"},
                        }},
                        {"args", {
                            {"type", "array"},
                            {"items", {{"type", "string"}}},
                            {"description", "Optional argv passed to the script after the script path"},
                        }},
                        {"timeout", {
                            {"type", "integer"},
                            {"minimum", 1},
                            {"maximum", SERVER_TOOL_RUN_PYTHON_MAX_TIMEOUT},
                            {"description", string_format("Timeout in seconds (default 30, max %d)", SERVER_TOOL_RUN_PYTHON_MAX_TIMEOUT)},
                        }},
                        {"max_output_size", {
                            {"type", "integer"},
                            {"minimum", 1},
                            {"maximum", (int) SERVER_TOOL_RUN_PYTHON_MAX_OUTPUT_SIZE},
                            {"description", string_format("Maximum output size in bytes (default %zu)", SERVER_TOOL_RUN_PYTHON_DEFAULT_OUTPUT_SIZE)},
                        }},
                    }},
                    {"additionalProperties", false},
                }},
            }},
        };
    }

    json invoke(json params) override {
        const std::string code = json_value(params, "code", std::string());
        const std::string path = json_value(params, "path", std::string());
        if (code.empty() == path.empty()) {
            return {{"error", "provide exactly one of \"code\" or \"path\""}};
        }

        int timeout = std::clamp(json_value(params, "timeout", 30), 1, SERVER_TOOL_RUN_PYTHON_MAX_TIMEOUT);
        size_t max_output = static_cast<size_t>(std::clamp(
            json_value(params, "max_output_size", (int) SERVER_TOOL_RUN_PYTHON_DEFAULT_OUTPUT_SIZE),
            1, (int) SERVER_TOOL_RUN_PYTHON_MAX_OUTPUT_SIZE));

        std::string script_path = path;
        std::string tmp_path;
        if (!code.empty()) {
            static std::atomic<int> counter{0};
            tmp_path = (fs::temp_directory_path() /
                ("llama_python_" + std::to_string(++counter) + ".py")).string();
            {
                std::ofstream f(tmp_path, std::ios::binary);
                if (!f) {
                    return {{"error", "failed to create temporary Python file"}};
                }
                f << code;
            }
            script_path = tmp_path;
        }

        std::vector<std::string> args = {
#ifdef _WIN32
            "python",
#else
            "python3",
#endif
            script_path,
        };
        if (params.contains("args") && params["args"].is_array()) {
            for (const auto & arg : params["args"]) {
                if (!arg.is_string()) {
                    if (!tmp_path.empty()) {
                        std::error_code ec;
                        fs::remove(tmp_path, ec);
                    }
                    return {{"error", "args must be an array of strings"}};
                }
                args.push_back(arg.get<std::string>());
            }
        }

        auto res = run_process(args, max_output, timeout);
#ifdef _WIN32
        if (res.exit_code != 0 && res.output.find("failed to spawn process") != std::string::npos) {
            args[0] = "py";
            res = run_process(args, max_output, timeout);
        }
#else
        if (res.exit_code != 0 && res.output.find("failed to spawn process") != std::string::npos) {
            args[0] = "python";
            res = run_process(args, max_output, timeout);
        }
#endif

        if (!tmp_path.empty()) {
            std::error_code ec;
            fs::remove(tmp_path, ec);
        }

        std::string text_output = res.output;
        text_output += string_format("\n[exit code: %d]", res.exit_code);
        if (res.timed_out) {
            text_output += " [exit due to timed out]";
        }
        return {{"plain_text_response", text_output}};
    }
};

//
// write_file: create or overwrite a file
//

struct server_tool_write_file : server_tool {
    server_tool_write_file() {
        name = "write_file";
        display_name = "Write file";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Write content to a file, creating it (including parent directories) if it does not exist. May use with edit_file for more complex edits."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",    {{"type", "string"}, {"description", "Path of the file to write"}}},
                        {"content", {{"type", "string"}, {"description", "Content to write"}}},
                    }},
                    {"required", json::array({"path", "content"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path    = params.at("path").get<std::string>();
        std::string content = params.at("content").get<std::string>();

        std::error_code ec;
        fs::path fpath(path);
        if (fpath.has_parent_path()) {
            fs::create_directories(fpath.parent_path(), ec);
            if (ec) {
                return {{"error", "failed to create directories: " + ec.message()}};
            }
        }

        std::ofstream f(path, std::ios::binary);
        if (!f) {
            return {{"error", "failed to open file for writing: " + path}};
        }
        f << content;
        if (!f) {
            return {{"error", "failed to write file: " + path}};
        }

        return {{"result", "file written successfully"}, {"path", path}, {"bytes", content.size()}};
    }
};

//
// edit_file: edit file content via line-based changes
//

struct server_tool_edit_file : server_tool {
    server_tool_edit_file() {
        name = "edit_file";
        display_name = "Edit file";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description",
                    "Edit a file by applying a list of line-based changes. "
                    "Each change targets a 1-based inclusive line range and has a mode: "
                    "\"replace\" (replace lines with content), "
                    "\"delete\" (remove lines, content must be empty string), "
                    "\"append\" (insert content after line_end). "
                    "Set line_start to -1 to target the end of file (line_end is ignored in that case). "
                    "Changes must not overlap. They are applied in reverse line order automatically."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",    {{"type", "string"}, {"description", "Path to the file to edit"}}},
                        {"changes", {
                            {"type", "array"},
                            {"description", "List of changes to apply"},
                            {"items", {
                                {"type", "object"},
                                {"properties", {
                                    {"mode",       {{"type", "string"},  {"description", "\"replace\", \"delete\", or \"append\""}}},
                                    {"line_start", {{"type", "integer"}, {"description", "First line of the range (1-based); use -1 for end of file"}}},
                                    {"line_end",   {{"type", "integer"}, {"description", "Last line of the range (1-based, inclusive); ignored when line_start is -1"}}},
                                    {"content",    {{"type", "string"},  {"description", "Content to insert; must be empty string for delete mode"}}},
                                }},
                                {"required", json::array({"mode", "line_start", "line_end", "content"})},
                            }},
                        }},
                    }},
                    {"required", json::array({"path", "changes"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path = params.at("path").get<std::string>();
        const json & changes = params.at("changes");

        if (!changes.is_array()) {
            return {{"error", "\"changes\" must be an array"}};
        }

        // read file into lines
        std::ifstream fin(path);
        if (!fin) {
            return {{"error", "failed to open file: " + path}};
        }
        std::vector<std::string> lines;
        {
            std::string line;
            while (std::getline(fin, line)) {
                lines.push_back(line);
            }
        }
        fin.close();

        // validate and collect changes, then sort descending by line_start
        struct change_entry {
            std::string mode;
            int line_start; // 1-based
            int line_end;   // 1-based inclusive
            std::string content;
        };
        std::vector<change_entry> entries;
        entries.reserve(changes.size());

        for (const auto & ch : changes) {
            change_entry e;
            e.mode       = ch.at("mode").get<std::string>();
            e.line_start = ch.at("line_start").get<int>();
            e.line_end   = ch.at("line_end").get<int>();
            e.content    = ch.at("content").get<std::string>();

            if (e.mode != "replace" && e.mode != "delete" && e.mode != "append") {
                return {{"error", "invalid mode \"" + e.mode + "\"; must be replace, delete, or append"}};
            }
            if (e.mode == "delete" && !e.content.empty()) {
                return {{"error", "content must be empty string for delete mode"}};
            }
            int n = (int) lines.size();
            if (e.line_start == -1) {
                // -1 targets end of file -> valid for append only; line_end is ignored
                if (e.mode != "append") {
                    return {{"error", "line_start -1 (end of file) is only valid for append mode"}};
                }
                // append at end of file: insert position is the current line count
                e.line_start = n;
                e.line_end   = n;
            } else {
                if (e.line_start < 1 || e.line_end < e.line_start) {
                    return {{"error", string_format("invalid line range [%d, %d]", e.line_start, e.line_end)}};
                }
                if (e.line_end > n) {
                    return {{"error", string_format("line_end %d exceeds file length %d", e.line_end, n)}};
                }
            }
            entries.push_back(std::move(e));
        }

        // sort descending so earlier-indexed changes don't shift later ones
        std::sort(entries.begin(), entries.end(), [](const change_entry & a, const change_entry & b) {
            return a.line_start > b.line_start;
        });

        // apply changes (0-based indices internally)
        for (const auto & e : entries) {
            int idx_start = e.line_start - 1; // 0-based
            int idx_end   = e.line_end   - 1; // 0-based inclusive

            // split content into lines (preserve trailing newline awareness)
            std::vector<std::string> new_lines;
            if (!e.content.empty()) {
                std::istringstream ss(e.content);
                std::string ln;
                while (std::getline(ss, ln)) {
                    new_lines.push_back(ln);
                }
                // if content ends with \n, getline consumed it — no extra empty line needed
                // if content does NOT end with \n, last line is still captured correctly
            }

            if (e.mode == "replace") {
                // erase [idx_start, idx_end] and insert new_lines
                lines.erase(lines.begin() + idx_start, lines.begin() + idx_end + 1);
                lines.insert(lines.begin() + idx_start, new_lines.begin(), new_lines.end());
            } else if (e.mode == "delete") {
                lines.erase(lines.begin() + idx_start, lines.begin() + idx_end + 1);
            } else { // append
                // insert after idx_end; idx_end + 1 == lines.size() for end-of-file append
                lines.insert(lines.begin() + (idx_end + 1), new_lines.begin(), new_lines.end());
            }
        }

        // write file back
        std::ofstream fout(path, std::ios::binary);
        if (!fout) {
            return {{"error", "failed to open file for writing: " + path}};
        }
        for (size_t i = 0; i < lines.size(); i++) {
            fout << lines[i];
            if (i + 1 < lines.size()) {
                fout << "\n";
            }
        }
        if (!lines.empty()) {
            fout << "\n";
        }
        if (!fout) {
            return {{"error", "failed to write file: " + path}};
        }

        return {{"result", "file edited successfully"}, {"path", path}, {"lines", (int) lines.size()}};
    }
};

//
// read-only Git tools
//

static constexpr size_t SERVER_TOOL_GIT_MAX_OUTPUT_SIZE = 64 * 1024;
static constexpr int    SERVER_TOOL_GIT_TIMEOUT         = 30;

static json run_git_tool(const std::string & repo_path, std::vector<std::string> args) {
    std::vector<std::string> command = {
        "git", "--no-pager", "-c", "core.pager=cat", "-c", "core.fsmonitor=false",
        "-c", "color.ui=false",
        "-C", repo_path,
    };
    command.insert(command.end(), args.begin(), args.end());

    const auto environment = sanitized_git_environment();
    const auto result = run_process(
        command, SERVER_TOOL_GIT_MAX_OUTPUT_SIZE, SERVER_TOOL_GIT_TIMEOUT, &environment);
    if (result.timed_out) {
        return {{"error", "git command timed out"}};
    }
    if (result.exit_code != 0) {
        return {{"error", "git command failed (exit " + std::to_string(result.exit_code) + "): " + result.output}};
    }
    return {{"plain_text_response", result.output.empty() ? "(no output)\n" : result.output}};
}

static bool invalid_revision(const std::string & revision) {
    return revision.empty() || revision[0] == '-';
}

struct server_tool_git_status : server_tool {
    server_tool_git_status() {
        name = "git_status";
        display_name = "Git status";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description",
                    "Inspect modified, staged, and untracked files without changing the repository. "
                    "Prefer this over exec_shell_command for git status."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"repo_path", {{"type", "string"}, {"description", "Repository path (default: current directory)"}}},
                    }},
                    {"additionalProperties", false},
                }},
            }},
        };
    }

    json invoke(json params) override {
        return run_git_tool(json_value(params, "repo_path", std::string(".")),
            {"status", "--short", "--branch", "--untracked-files=all"});
    }
};

struct server_tool_git_diff : server_tool {
    server_tool_git_diff() {
        name = "git_diff";
        display_name = "Git diff";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description",
                    "Show Git working-tree or staged diffs. With staged=false (default), only unstaged changes are returned. "
                    "Prefer this over shelling out to git diff."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"repo_path", {{"type", "string"},  {"description", "Repository path (default: current directory)"}}},
                        {"staged",    {{"type", "boolean"}, {"description", "If true, show staged changes only (default: false)"}}},
                        {"path",      {{"type", "string"},  {"description", "Optional path filter; values starting with '-' are treated as paths after --"}}},
                    }},
                    {"additionalProperties", false},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::vector<std::string> args = {"diff", "--no-ext-diff", "--no-textconv"};
        if (json_value(params, "staged", false)) {
            args.push_back("--cached");
        }
        const std::string path = json_value(params, "path", std::string());
        if (!path.empty()) {
            args.push_back("--");
            args.push_back(path);
        }
        return run_git_tool(json_value(params, "repo_path", std::string(".")), std::move(args));
    }
};

struct server_tool_git_log : server_tool {
    server_tool_git_log() {
        name = "git_log";
        display_name = "Git log";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description",
                    "List recent commits, optionally filtered by path. Prefer this over shelling out to git log."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"repo_path", {{"type", "string"},  {"description", "Repository path (default: current directory)"}}},
                        {"max_count", {{"type", "integer"}, {"minimum", 1}, {"maximum", 100}, {"description", "Maximum commits (default 20, max 100)"}}},
                        {"path",      {{"type", "string"},  {"description", "Optional path filter"}}},
                    }},
                    {"additionalProperties", false},
                }},
            }},
        };
    }

    json invoke(json params) override {
        const int max_count = std::clamp(json_value(params, "max_count", 20), 1, 100);
        std::vector<std::string> args = {
            "log", "--date=iso-strict", "--format=%H%nAuthor: %an <%ae>%nDate: %ad%nSubject: %s%n",
            "--max-count=" + std::to_string(max_count),
        };
        const std::string path = json_value(params, "path", std::string());
        if (!path.empty()) {
            args.push_back("--");
            args.push_back(path);
        }
        return run_git_tool(json_value(params, "repo_path", std::string(".")), std::move(args));
    }
};

struct server_tool_git_show : server_tool {
    server_tool_git_show() {
        name = "git_show";
        display_name = "Git show";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description",
                    "Show a commit (message and patch) or the contents of a file at a revision. "
                    "Provide path to read a file at that revision; omit path to inspect the commit itself."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"repo_path", {{"type", "string"}, {"description", "Repository path (default: current directory)"}}},
                        {"revision",  {{"type", "string"}, {"description", "Revision to show (default: HEAD); must not begin with '-'"}}},
                        {"path",      {{"type", "string"}, {"description", "Optional file path at the revision"}}},
                    }},
                    {"additionalProperties", false},
                }},
            }},
        };
    }

    json invoke(json params) override {
        const std::string revision = json_value(params, "revision", std::string("HEAD"));
        if (invalid_revision(revision)) {
            return {{"error", "revision must not be empty or begin with '-'"}};
        }
        const std::string path = json_value(params, "path", std::string());
        std::vector<std::string> args;
        if (path.empty()) {
            args = {"show", "--no-ext-diff", "--no-textconv", "--format=fuller", revision};
        } else {
            args = {"show", "--no-ext-diff", "--no-textconv", revision + ":" + path};
        }
        return run_git_tool(json_value(params, "repo_path", std::string(".")), std::move(args));
    }
};

struct server_tool_git_blame : server_tool {
    server_tool_git_blame() {
        name = "git_blame";
        display_name = "Git blame";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description",
                    "Identify the commit and author responsible for each line in a file. "
                    "Use start_line/end_line together to limit the range."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"repo_path", {{"type", "string"},  {"description", "Repository path (default: current directory)"}}},
                        {"path",      {{"type", "string"},  {"description", "File path"}}},
                        {"revision",  {{"type", "string"},  {"description", "Revision to blame (default: HEAD); must not begin with '-'"}}},
                        {"start_line",{{"type", "integer"}, {"minimum", 1}, {"description", "Optional first line, 1-based"}}},
                        {"end_line",  {{"type", "integer"}, {"minimum", 1}, {"description", "Optional last line, 1-based"}}},
                    }},
                    {"required", json::array({"path"})},
                    {"additionalProperties", false},
                }},
            }},
        };
    }

    json invoke(json params) override {
        const std::string revision = json_value(params, "revision", std::string("HEAD"));
        if (invalid_revision(revision)) {
            return {{"error", "revision must not be empty or begin with '-'"}};
        }
        const int start_line = json_value(params, "start_line", 0);
        const int end_line = json_value(params, "end_line", 0);
        if ((start_line == 0) != (end_line == 0) || start_line < 0 || end_line < start_line) {
            return {{"error", "start_line and end_line must be provided together as a valid range"}};
        }

        std::vector<std::string> args = {"blame", "--date=iso-strict"};
        if (start_line > 0) {
            args.push_back("-L");
            args.push_back(std::to_string(start_line) + "," + std::to_string(end_line));
        }
        args.push_back(revision);
        args.push_back("--");
        args.push_back(params.at("path").get<std::string>());
        return run_git_tool(json_value(params, "repo_path", std::string(".")), std::move(args));
    }
};

//
// apply_diff: apply a unified diff via git apply
//

struct server_tool_apply_diff : server_tool {
    server_tool_apply_diff() {
        name = "apply_diff";
        display_name = "Apply diff";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Apply a unified diff to edit one or more files using git apply. Use this instead of edit_file when the changes are complex."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"diff", {{"type", "string"}, {"description", "Unified diff content in git diff format"}}},
                    }},
                    {"required", json::array({"diff"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string diff = params.at("diff").get<std::string>();

        // write diff to a temporary file
        static std::atomic<int> counter{0};
        std::string tmp_path = (fs::temp_directory_path() /
            ("llama_patch_" + std::to_string(++counter) + ".patch")).string();

        {
            std::ofstream f(tmp_path, std::ios::binary);
            if (!f) {
                return {{"error", "failed to create temp patch file"}};
            }
            f << diff;
        }

        auto res = run_process({"git", "apply", tmp_path}, 4096, 10);

        std::error_code ec;
        fs::remove(tmp_path, ec);

        if (res.exit_code != 0) {
            return {{"error", "git apply failed (exit " + std::to_string(res.exit_code) + "): " + res.output}};
        }
        return {{"result", "patch applied successfully"}};
    }
};

//
// get_datetime: returns the current date and time
//

struct server_tool_get_datetime : server_tool {
    server_tool_get_datetime() {
        name = "get_datetime";
        display_name = "Get Date & Time";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Returns the current date and time"},
            }},
        };
    }

    json invoke(json) override {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);

        return {{"result", std::ctime(&time)}};
    }
};

//
// public API
//

static std::vector<std::unique_ptr<server_tool>> build_tools() {
    std::vector<std::unique_ptr<server_tool>> tools;
    tools.push_back(std::make_unique<server_tool_read_file>());
    tools.push_back(std::make_unique<server_tool_file_glob_search>());
    tools.push_back(std::make_unique<server_tool_grep_search>());
    tools.push_back(std::make_unique<server_tool_web_search>());
    tools.push_back(std::make_unique<server_tool_fetch_url>());
    tools.push_back(std::make_unique<server_tool_exec_shell_command>());
    tools.push_back(std::make_unique<server_tool_run_python>());
    tools.push_back(std::make_unique<server_tool_write_file>());
    tools.push_back(std::make_unique<server_tool_edit_file>());
    tools.push_back(std::make_unique<server_tool_git_status>());
    tools.push_back(std::make_unique<server_tool_git_diff>());
    tools.push_back(std::make_unique<server_tool_git_log>());
    tools.push_back(std::make_unique<server_tool_git_show>());
    tools.push_back(std::make_unique<server_tool_git_blame>());
    tools.push_back(std::make_unique<server_tool_apply_diff>());
    tools.push_back(std::make_unique<server_tool_get_datetime>());
    return tools;
}

void server_tools::setup(const std::vector<std::string> & enabled_tools) {
    if (!enabled_tools.empty()) {
        std::unordered_set<std::string> enabled_set(enabled_tools.begin(), enabled_tools.end());
        auto all_tools = build_tools();

        // collect all known tool names for validation
        std::vector<std::string> known_names;
        known_names.reserve(all_tools.size());
        for (const auto & t : all_tools) {
            known_names.push_back(t->name);
        }

        // validate that every requested tool is known
        for (const auto & name : enabled_tools) {
            if (name == "all") continue;
            if (std::find(known_names.begin(), known_names.end(), name) == known_names.end()) {
                throw std::runtime_error(string_format(
                    "unknown tool \"%s\". available tools: %s",
                    name.c_str(),
                    string_join(known_names, ", ").c_str()));
            }
        }

        tools.clear();
        for (auto & t : all_tools) {
            if (enabled_set.count(t->name) > 0 || enabled_set.count("all") > 0) {
                tools.push_back(std::move(t));
            }
        }
    }

    handle_get = [this](const server_http_req &) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        try {
            json result = json::array();
            for (const auto & t : tools) {
                result.push_back(t->to_json());
            }
            res->data = safe_json_to_str(result);
        } catch (const std::exception & e) {
            SRV_ERR("got exception: %s\n", e.what());
            res->status = 500;
            res->data   = safe_json_to_str(format_error_response(e.what(), ERROR_TYPE_SERVER));
        }
        return res;
    };

    handle_post = [this](const server_http_req & req) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        try {
            json body = json::parse(req.body);
            std::string tool_name = body.at("tool").get<std::string>();
            json params = body.value("params", json::object());
            json result = invoke(tool_name, params);
            res->data   = safe_json_to_str(result);
        } catch (const json::exception & e) {
            res->status = 400;
            res->data   = safe_json_to_str(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        } catch (const std::exception & e) {
            SRV_ERR("got exception: %s\n", e.what());
            res->status = 500;
            res->data   = safe_json_to_str(format_error_response(e.what(), ERROR_TYPE_SERVER));
        }
        return res;
    };
}

json server_tools::invoke(const std::string & name, const json & params) {
    for (auto & t : tools) {
        if (t->name == name) {
            return t->invoke(params);
        }
    }
    return {{"error", "unknown tool: " + name}};
}
