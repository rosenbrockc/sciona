import fs from "node:fs/promises";
import net from "node:net";
import path from "node:path";
import process from "node:process";
import readline from "node:readline";
import { pathToFileURL } from "node:url";

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    if (!key.startsWith("--")) {
      continue;
    }
    const name = key.slice(2);
    const value = argv[i + 1];
    if (!value || value.startsWith("--")) {
      args[name] = "true";
      continue;
    }
    args[name] = value;
    i += 1;
  }
  return args;
}

function jsonRpcResult(id, result) {
  return JSON.stringify({ jsonrpc: "2.0", id, result }) + "\n";
}

function jsonRpcError(id, code, message, data = null) {
  return JSON.stringify({
    jsonrpc: "2.0",
    id,
    error: { code, message, data },
  }) + "\n";
}

function extractText(response) {
  const candidates = response?.candidates;
  if (!Array.isArray(candidates)) {
    return "";
  }
  const parts = [];
  for (const candidate of candidates) {
    const contentParts = candidate?.content?.parts;
    if (!Array.isArray(contentParts)) {
      continue;
    }
    for (const part of contentParts) {
      if (typeof part?.text === "string" && !part?.thought) {
        parts.push(part.text);
      }
    }
  }
  return parts.join("").trim();
}

async function loadGeminiRuntime(cliRoot, model, cwd) {
  const configMod = await import(
    pathToFileURL(path.join(cliRoot, "dist/src/config/config.js")).href
  );
  const settingsMod = await import(
    pathToFileURL(path.join(cliRoot, "dist/src/config/settings.js")).href
  );
  const authMod = await import(
    pathToFileURL(path.join(cliRoot, "dist/src/validateNonInterActiveAuth.js")).href
  );

  const settings = settingsMod.loadSettings(cwd);
  const argv = {
    prompt: "ageom-shim-init",
    query: undefined,
    promptInteractive: undefined,
    experimentalAcp: false,
    isCommand: false,
    sandbox: false,
    yolo: false,
    approvalMode: "default",
    policy: undefined,
    allowedMcpServerNames: undefined,
    allowedTools: [],
    extensions: undefined,
    listExtensions: false,
    resume: undefined,
    listSessions: false,
    deleteSession: undefined,
    includeDirectories: undefined,
    screenReader: false,
    outputFormat: "text",
    debug: false,
    model,
  };

  const sessionId = `ageom-shim-${process.pid}`;
  const config = await configMod.loadCliConfig(settings.merged, sessionId, argv, {
    cwd,
  });
  const authType = await authMod.validateNonInteractiveAuth(
    settings.merged.security.auth.selectedType,
    settings.merged.security.auth.useExternal,
    config,
    settings,
  );
  await config.refreshAuth(authType);
  await config.initialize();
  return config.getBaseLlmClient();
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const socketPath = args.socket;
  const model = args.model || "flash-lite";
  const cwd = args.cwd || process.cwd();
  const cliRoot = process.env.AGEOM_GEMINI_CLI_ROOT || "";
  const fakeMode = process.env.AGEOM_GEMINI_DAEMON_FAKE === "1";

  if (!socketPath) {
    throw new Error("--socket is required");
  }

  process.chdir(cwd);
  try {
    await fs.rm(socketPath, { force: true });
  } catch {
    // ignore stale socket cleanup failures
  }

  let requestCount = 0;
  let baseLlmClient = null;
  let firstCompletion = true;

  if (!fakeMode) {
    if (!cliRoot) {
      throw new Error("AGEOM_GEMINI_CLI_ROOT is not set");
    }
    baseLlmClient = await loadGeminiRuntime(cliRoot, model, cwd);
  }

  const server = net.createServer((socket) => {
    socket.setEncoding("utf8");
    const rl = readline.createInterface({ input: socket, crlfDelay: Infinity });

    rl.on("line", async (line) => {
      let message;
      try {
        message = JSON.parse(line);
      } catch (error) {
        socket.write(jsonRpcError(null, -32700, "Parse error", String(error)));
        return;
      }

      const { id, method, params = {} } = message;
      try {
        if (method === "ping") {
          socket.write(
            jsonRpcResult(id, { ok: true, pid: process.pid, model, requestCount }),
          );
          return;
        }

        if (method !== "complete") {
          socket.write(jsonRpcError(id, -32601, "Method not found", method));
          return;
        }

        requestCount += 1;
        const system = typeof params.system === "string" ? params.system : "";
        const user = typeof params.user === "string" ? params.user : "";

        let text;
        if (fakeMode) {
          text = `fake pid=${process.pid} count=${requestCount} model=${model} system=${system} user=${user}`;
        } else {
          const response = await baseLlmClient.generateContent({
            modelConfigKey: { model },
            contents: [{ role: "user", parts: [{ text: user }] }],
            systemInstruction: system || undefined,
            promptId: `ageom-shim-${process.pid}-${requestCount}`,
          });
          text = extractText(response);
        }

        socket.write(
          jsonRpcResult(id, {
            text,
            pid: process.pid,
            requestCount,
            model,
            coldStart: firstCompletion,
          }),
        );
        firstCompletion = false;
      } catch (error) {
        socket.write(
          jsonRpcError(
            id,
            -32000,
            error instanceof Error ? error.message : String(error),
          ),
        );
      }
    });

    socket.on("error", () => {
      rl.close();
    });
  });

  const cleanup = async () => {
    server.close();
    try {
      await fs.rm(socketPath, { force: true });
    } catch {
      // ignore cleanup failures
    }
  };

  process.on("SIGINT", async () => {
    await cleanup();
    process.exit(0);
  });
  process.on("SIGTERM", async () => {
    await cleanup();
    process.exit(0);
  });

  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(socketPath, () => resolve());
  });
}

main().catch((error) => {
  const message = error instanceof Error ? error.stack || error.message : String(error);
  process.stderr.write(`${message}\n`);
  process.exit(1);
});
