import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
} from "vscode-languageclient/node";

let client: LanguageClient | undefined;

export async function activate(): Promise<void> {
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

export async function deactivate(): Promise<void> {
  await client?.stop();
}

function serverPath(): string {
  const configured = vscode.workspace
    .getConfiguration("yuzu")
    .get<string>("server.path");
  if (configured) {
    return configured;
  }

  for (const folder of vscode.workspace.workspaceFolders ?? []) {
    const built = path.join(folder.uri.fsPath, "target", "debug", "yuzu-lsp");
    if (fs.existsSync(built)) {
      return built;
    }
  }
  return "yuzu-lsp";
}
