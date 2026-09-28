import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
} from "vscode-languageclient/node";

let client: LanguageClient | undefined;

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  context.subscriptions.push(
    vscode.workspace.onDidGrantWorkspaceTrust(() => restart()),
    { dispose: () => void client?.stop() },
  );
  await start();
}

export async function deactivate(): Promise<void> {
  await client?.stop();
}

async function start(): Promise<void> {
  const serverOptions: ServerOptions = { command: serverPath() };
  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ language: "yuzu" }],
  };

  client = new LanguageClient("yuzu", "Yuzu", serverOptions, clientOptions);
  try {
    await client.start();
  } catch (error) {
    client = undefined;
    void vscode.window.showErrorMessage(
      `Yuzu: the language server did not start (${String(error)}). ` +
        "Build it with `cargo build -p yuzu_lsp`, or set `yuzu.server.path`.",
    );
  }
}

async function restart(): Promise<void> {
  await client?.stop();
  await start();
}

// In an untrusted workspace the setting comes from the user's settings only,
// and the workspace's own build is not run.
function serverPath(): string {
  const configured = vscode.workspace
    .getConfiguration("yuzu")
    .get<string>("server.path");
  if (configured) {
    return configured;
  }

  const binary = process.platform === "win32" ? "yuzu-lsp.exe" : "yuzu-lsp";
  if (vscode.workspace.isTrusted) {
    const built = (vscode.workspace.workspaceFolders ?? [])
      .flatMap((folder) =>
        ["debug", "release"].map((profile) =>
          path.join(folder.uri.fsPath, "target", profile, binary),
        ),
      )
      .filter((candidate) => fs.existsSync(candidate))
      .sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);
    if (built.length > 0) {
      return built[0];
    }
  }
  return binary;
}
