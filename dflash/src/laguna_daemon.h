// Laguna daemon library entry point.
//
// Encapsulates the stdin/stream-fd protocol for the Poolside Laguna-XS.2
// target so a single daemon binary (test_dflash) can dispatch by GGUF arch
// without duplicating the daemon loop. The qwen35 stack stays in test_dflash;
// laguna requests funnel through run_laguna_daemon().
//
// Wire format mirrors what scripts/server.py'́s _build_cmd_line() emits for
// the qwen35 stack — see the body of run_laguna_daemon for the supported
// commands. SNAPSHOT/RESTORE/FREE_SNAPSHOT and PFlash compress/park/unpark
// are not yet implemented; server.py forces prefix-cache slots to 0 and
// disables PFlash compression on the laguna path until they land.

#pragma once

#include <string>
#include "ggml.h"

namespace dflash27b {

struct LagunaDaemonArgs {
    std::string target_path;       // path to laguna-*.gguf
    int         max_ctx   = 16384; // K/V cache capacity in tokens
    int         chunk     = 2048;  // chunked-prefill chunk size
    ggml_type   kv_type   = GGML_TYPE_Q8_0;
    int         stream_fd = -1;    // server.py's writable pipe end (int32 LE
                                   // tokens, terminated by -1 sentinel). -1
                                   // means the bare-prompt protocol is
                                   // disabled and only the legacy `generate`
                                   // file-out command is accepted.
};

// Boots the laguna target on a fresh CUDA backend, prints a `[laguna-daemon]
// ready ...` banner on stdout, and services stdin commands until `quit`,
// `exit`, or EOF. Returns the process exit code (0 on clean shutdown).
int run_laguna_daemon(const LagunaDaemonArgs & args);

}  // namespace dflash27b
