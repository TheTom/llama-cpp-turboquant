#ifdef NDEBUG
#undef NDEBUG
#endif

#include "server-tools.h"

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

static const json & find_tool(const server_tools & tools, const std::string & name, json & storage) {
    for (const auto & tool : tools.tools) {
        if (tool->name == name) {
            storage = tool->to_json();
            return storage;
        }
    }
    assert(false);
    return storage;
}

static void test_definitions() {
    server_tools tools;
    tools.setup({"web_search", "fetch_url", "git_status", "git_diff", "git_log", "git_show", "git_blame"});
    assert(tools.tools.size() == 7);

    for (const auto & tool : tools.tools) {
        const json definition = tool->to_json();
        assert(definition["type"] == "builtin");
        assert(definition["permissions"]["write"] == false);
        assert(definition["definition"]["type"] == "function");
        assert(definition["definition"]["function"]["name"] == tool->name);
    }

    json storage;
    const json & fetch = find_tool(tools, "fetch_url", storage);
    assert(fetch["definition"]["function"]["parameters"]["required"] == json::array({"url"}));
}

static void test_validation() {
    server_tools tools;
    tools.setup({"web_search", "fetch_url", "run_python", "git_show", "git_blame"});

    const json empty_query = tools.invoke("web_search", {{"query", ""}});
    assert(empty_query.contains("error"));

    const json missing_python = tools.invoke("run_python", json::object());
    assert(missing_python.contains("error"));

    const json both_python = tools.invoke("run_python", {{"code", "print(1)"}, {"path", "x.py"}});
    assert(both_python.contains("error"));

    const json private_url = tools.invoke("fetch_url", {{"url", "http://127.0.0.1:9090/"}});
    assert(private_url.contains("error"));
    assert(private_url["error"].get<std::string>().find("private or reserved") != std::string::npos);

    const json credentials = tools.invoke("fetch_url", {{"url", "https://user:pass@example.com/"}});
    assert(credentials.contains("error"));

    const json bad_revision = tools.invoke("git_show", {{"revision", "--help"}});
    assert(bad_revision.contains("error"));

    const json bad_range = tools.invoke("git_blame", {
        {"path", "file.txt"},
        {"start_line", 2},
    });
    assert(bad_range.contains("error"));
}

static int run(const std::string & command) {
    return std::system(command.c_str());
}

static void test_git_tools() {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path repo = fs::temp_directory_path() /
        ("llama-server-tools-test-" + std::to_string(suffix));
    std::error_code ec;
    fs::remove_all(repo, ec);
    fs::create_directories(repo);

    const std::string quoted = "\"" + repo.string() + "\"";
    assert(run("git init -q " + quoted) == 0);
    assert(run("git -C " + quoted + " config user.name \"Test User\"") == 0);
    assert(run("git -C " + quoted + " config user.email test@example.com") == 0);

    {
        std::ofstream file(repo / "file.txt");
        file << "first\nsecond\n";
    }
    assert(run("git -C " + quoted + " add -- file.txt") == 0);
    assert(run("git -C " + quoted + " commit -q -m initial") == 0);

    {
        std::ofstream file(repo / "file.txt", std::ios::app);
        file << "third\n";
    }

    server_tools tools;
    tools.setup({"git_status", "git_diff", "git_log", "git_show", "git_blame"});
    const json params = {{"repo_path", repo.string()}};

    const json status = tools.invoke("git_status", params);
    assert(status.contains("plain_text_response"));
    assert(status["plain_text_response"].get<std::string>().find("file.txt") != std::string::npos);

    const json diff = tools.invoke("git_diff", params);
    assert(diff.contains("plain_text_response"));
    assert(diff["plain_text_response"].get<std::string>().find("+third") != std::string::npos);

    const json log = tools.invoke("git_log", params);
    assert(log.contains("plain_text_response"));
    assert(log["plain_text_response"].get<std::string>().find("Subject: initial") != std::string::npos);

    const json show = tools.invoke("git_show", {{"repo_path", repo.string()}, {"revision", "HEAD"}, {"path", "file.txt"}});
    assert(show.contains("plain_text_response"));
    assert(show["plain_text_response"].get<std::string>().find("second") != std::string::npos);

    const json blame = tools.invoke("git_blame", {{"repo_path", repo.string()}, {"path", "file.txt"}});
    assert(blame.contains("plain_text_response"));
    assert(blame["plain_text_response"].get<std::string>().find("first") != std::string::npos);

    const json injected_path = tools.invoke("git_diff", {{"repo_path", repo.string()}, {"path", "--stat"}});
    assert(injected_path.contains("plain_text_response"));

    fs::remove_all(repo, ec);
}

int main() {
    test_definitions();
    test_validation();
    test_git_tools();
    return 0;
}
