/**
 * NanoClaw Agent Runner
 * Runs inside a container, receives config via stdin, outputs result to stdout
 *
 * Input protocol:
 *   Stdin: Full ContainerInput JSON (read until EOF, like before)
 *   IPC:   Follow-up messages written as JSON files to /workspace/ipc/input/
 *          Files: {type:"message", text:"..."}.json — polled and consumed
 *          Sentinel: /workspace/ipc/input/_close — signals session end
 *
 * Stdout protocol:
 *   Each result is wrapped in OUTPUT_START_MARKER / OUTPUT_END_MARKER pairs.
 *   Multiple results may be emitted (one per agent teams result).
 *   Final marker after loop ends signals completion.
 */

import fs from 'fs';
import path from 'path';
import { query, HookCallback, PreCompactHookInput, PreToolUseHookInput } from '@anthropic-ai/claude-agent-sdk';
import { fileURLToPath } from 'url';

interface ContainerInput {
  prompt: string;
  sessionId?: string;
  groupFolder: string;
  chatJid: string;
  isMain: boolean;
  isScheduledTask?: boolean;
  assistantName?: string;
  secrets?: Record<string, string>;
}

interface ContainerOutput {
  status: 'success' | 'error';
  result: string | null;
  newSessionId?: string;
  error?: string;
}

interface SessionEntry {
  sessionId: string;
  fullPath: string;
  summary: string;
  firstPrompt: string;
}

interface GeminiHistoryMessage {
  role: 'user' | 'model';
  text: string;
}

interface SessionsIndex {
  entries: SessionEntry[];
}

interface SDKUserMessage {
  type: 'user';
  message: { role: 'user'; content: string };
  parent_tool_use_id: null;
  session_id: string;
}

const IPC_INPUT_DIR = '/workspace/ipc/input';
const IPC_INPUT_CLOSE_SENTINEL = path.join(IPC_INPUT_DIR, '_close');
const IPC_POLL_MS = 500;
const DEFAULT_GEMINI_MODEL = 'gemini-2.5-flash-lite';
const geminiHistoryBySession = new Map<string, GeminiHistoryMessage[]>();
const DEFAULT_WEB_RESULTS = 6;

/**
 * Push-based async iterable for streaming user messages to the SDK.
 * Keeps the iterable alive until end() is called, preventing isSingleUserTurn.
 */
class MessageStream {
  private queue: SDKUserMessage[] = [];
  private waiting: (() => void) | null = null;
  private done = false;

  push(text: string): void {
    this.queue.push({
      type: 'user',
      message: { role: 'user', content: text },
      parent_tool_use_id: null,
      session_id: '',
    });
    this.waiting?.();
  }

  end(): void {
    this.done = true;
    this.waiting?.();
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<SDKUserMessage> {
    while (true) {
      while (this.queue.length > 0) {
        yield this.queue.shift()!;
      }
      if (this.done) return;
      await new Promise<void>(r => { this.waiting = r; });
      this.waiting = null;
    }
  }
}

async function readStdin(): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', chunk => { data += chunk; });
    process.stdin.on('end', () => resolve(data));
    process.stdin.on('error', reject);
  });
}

const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
const OUTPUT_END_MARKER = '---NANOCLAW_OUTPUT_END---';

function writeOutput(output: ContainerOutput): void {
  console.log(OUTPUT_START_MARKER);
  console.log(JSON.stringify(output));
  console.log(OUTPUT_END_MARKER);
}

function log(message: string): void {
  console.error(`[agent-runner] ${message}`);
}

function getSessionSummary(sessionId: string, transcriptPath: string): string | null {
  const projectDir = path.dirname(transcriptPath);
  const indexPath = path.join(projectDir, 'sessions-index.json');

  if (!fs.existsSync(indexPath)) {
    log(`Sessions index not found at ${indexPath}`);
    return null;
  }

  try {
    const index: SessionsIndex = JSON.parse(fs.readFileSync(indexPath, 'utf-8'));
    const entry = index.entries.find(e => e.sessionId === sessionId);
    if (entry?.summary) {
      return entry.summary;
    }
  } catch (err) {
    log(`Failed to read sessions index: ${err instanceof Error ? err.message : String(err)}`);
  }

  return null;
}

/**
 * Archive the full transcript to conversations/ before compaction.
 */
function createPreCompactHook(assistantName?: string): HookCallback {
  return async (input, _toolUseId, _context) => {
    const preCompact = input as PreCompactHookInput;
    const transcriptPath = preCompact.transcript_path;
    const sessionId = preCompact.session_id;

    if (!transcriptPath || !fs.existsSync(transcriptPath)) {
      log('No transcript found for archiving');
      return {};
    }

    try {
      const content = fs.readFileSync(transcriptPath, 'utf-8');
      const messages = parseTranscript(content);

      if (messages.length === 0) {
        log('No messages to archive');
        return {};
      }

      const summary = getSessionSummary(sessionId, transcriptPath);
      const name = summary ? sanitizeFilename(summary) : generateFallbackName();

      const conversationsDir = '/workspace/group/conversations';
      fs.mkdirSync(conversationsDir, { recursive: true });

      const date = new Date().toISOString().split('T')[0];
      const filename = `${date}-${name}.md`;
      const filePath = path.join(conversationsDir, filename);

      const markdown = formatTranscriptMarkdown(messages, summary, assistantName);
      fs.writeFileSync(filePath, markdown);

      log(`Archived conversation to ${filePath}`);
    } catch (err) {
      log(`Failed to archive transcript: ${err instanceof Error ? err.message : String(err)}`);
    }

    return {};
  };
}

// Secrets to strip from Bash tool subprocess environments.
// These are needed by claude-code for API auth but should never
// be visible to commands Kit runs.
const SECRET_ENV_VARS = ['ANTHROPIC_API_KEY', 'CLAUDE_CODE_OAUTH_TOKEN'];

function createSanitizeBashHook(): HookCallback {
  return async (input, _toolUseId, _context) => {
    const preInput = input as PreToolUseHookInput;
    const command = (preInput.tool_input as { command?: string })?.command;
    if (!command) return {};

    const unsetPrefix = `unset ${SECRET_ENV_VARS.join(' ')} 2>/dev/null; `;
    return {
      hookSpecificOutput: {
        hookEventName: 'PreToolUse',
        updatedInput: {
          ...(preInput.tool_input as Record<string, unknown>),
          command: unsetPrefix + command,
        },
      },
    };
  };
}

function sanitizeFilename(summary: string): string {
  return summary
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 50);
}

function generateFallbackName(): string {
  const time = new Date();
  return `conversation-${time.getHours().toString().padStart(2, '0')}${time.getMinutes().toString().padStart(2, '0')}`;
}

interface ParsedMessage {
  role: 'user' | 'assistant';
  content: string;
}

function parseTranscript(content: string): ParsedMessage[] {
  const messages: ParsedMessage[] = [];

  for (const line of content.split('\n')) {
    if (!line.trim()) continue;
    try {
      const entry = JSON.parse(line);
      if (entry.type === 'user' && entry.message?.content) {
        const text = typeof entry.message.content === 'string'
          ? entry.message.content
          : entry.message.content.map((c: { text?: string }) => c.text || '').join('');
        if (text) messages.push({ role: 'user', content: text });
      } else if (entry.type === 'assistant' && entry.message?.content) {
        const textParts = entry.message.content
          .filter((c: { type: string }) => c.type === 'text')
          .map((c: { text: string }) => c.text);
        const text = textParts.join('');
        if (text) messages.push({ role: 'assistant', content: text });
      }
    } catch {
    }
  }

  return messages;
}

function formatTranscriptMarkdown(messages: ParsedMessage[], title?: string | null, assistantName?: string): string {
  const now = new Date();
  const formatDateTime = (d: Date) => d.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    hour12: true
  });

  const lines: string[] = [];
  lines.push(`# ${title || 'Conversation'}`);
  lines.push('');
  lines.push(`Archived: ${formatDateTime(now)}`);
  lines.push('');
  lines.push('---');
  lines.push('');

  for (const msg of messages) {
    const sender = msg.role === 'user' ? 'User' : (assistantName || 'Assistant');
    const content = msg.content.length > 2000
      ? msg.content.slice(0, 2000) + '...'
      : msg.content;
    lines.push(`**${sender}**: ${content}`);
    lines.push('');
  }

  return lines.join('\n');
}

/**
 * Check for _close sentinel.
 */
function shouldClose(): boolean {
  if (fs.existsSync(IPC_INPUT_CLOSE_SENTINEL)) {
    try { fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL); } catch { /* ignore */ }
    return true;
  }
  return false;
}

/**
 * Drain all pending IPC input messages.
 * Returns messages found, or empty array.
 */
function drainIpcInput(): string[] {
  try {
    fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });
    const files = fs.readdirSync(IPC_INPUT_DIR)
      .filter(f => f.endsWith('.json'))
      .sort();

    const messages: string[] = [];
    for (const file of files) {
      const filePath = path.join(IPC_INPUT_DIR, file);
      try {
        const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        fs.unlinkSync(filePath);
        if (data.type === 'message' && data.text) {
          messages.push(data.text);
        }
      } catch (err) {
        log(`Failed to process input file ${file}: ${err instanceof Error ? err.message : String(err)}`);
        try { fs.unlinkSync(filePath); } catch { /* ignore */ }
      }
    }
    return messages;
  } catch (err) {
    log(`IPC drain error: ${err instanceof Error ? err.message : String(err)}`);
    return [];
  }
}

/**
 * Wait for a new IPC message or _close sentinel.
 * Returns the messages as a single string, or null if _close.
 */
function waitForIpcMessage(): Promise<string | null> {
  return new Promise((resolve) => {
    const poll = () => {
      if (shouldClose()) {
        resolve(null);
        return;
      }
      const messages = drainIpcInput();
      if (messages.length > 0) {
        resolve(messages.join('\n'));
        return;
      }
      setTimeout(poll, IPC_POLL_MS);
    };
    poll();
  });
}

/**
 * Run a single query and stream results via writeOutput.
 * Uses MessageStream (AsyncIterable) to keep isSingleUserTurn=false,
 * allowing agent teams subagents to run to completion.
 * Also pipes IPC messages into the stream during the query.
 */
async function runQuery(
  prompt: string,
  sessionId: string | undefined,
  mcpServerPath: string,
  containerInput: ContainerInput,
  sdkEnv: Record<string, string | undefined>,
  resumeAt?: string,
): Promise<{ newSessionId?: string; lastAssistantUuid?: string; closedDuringQuery: boolean }> {
  const stream = new MessageStream();
  stream.push(prompt);

  // Poll IPC for follow-up messages and _close sentinel during the query
  let ipcPolling = true;
  let closedDuringQuery = false;
  const pollIpcDuringQuery = () => {
    if (!ipcPolling) return;
    if (shouldClose()) {
      log('Close sentinel detected during query, ending stream');
      closedDuringQuery = true;
      stream.end();
      ipcPolling = false;
      return;
    }
    const messages = drainIpcInput();
    for (const text of messages) {
      log(`Piping IPC message into active query (${text.length} chars)`);
      stream.push(text);
    }
    setTimeout(pollIpcDuringQuery, IPC_POLL_MS);
  };
  setTimeout(pollIpcDuringQuery, IPC_POLL_MS);

  let newSessionId: string | undefined;
  let lastAssistantUuid: string | undefined;
  let messageCount = 0;
  let resultCount = 0;

  // Load global CLAUDE.md as additional system context (shared across all groups)
  const globalClaudeMdPath = '/workspace/global/CLAUDE.md';
  let globalClaudeMd: string | undefined;
  if (!containerInput.isMain && fs.existsSync(globalClaudeMdPath)) {
    globalClaudeMd = fs.readFileSync(globalClaudeMdPath, 'utf-8');
  }
  // Load SOUL.md as global identity context (applies to all groups when mounted)
  const soulMdPath = '/workspace/SOUL.md';
  let soulMd: string | undefined;
  if (fs.existsSync(soulMdPath)) {
    soulMd = fs.readFileSync(soulMdPath, 'utf-8');
  }
  const systemPromptParts: string[] = [];
  if (soulMd) systemPromptParts.push(soulMd);
  if (globalClaudeMd) systemPromptParts.push(globalClaudeMd);
  if (systemPromptParts.length > 0) {
    log(`System context loaded from: ${[
      soulMd ? 'SOUL.md' : null,
      globalClaudeMd ? 'global/CLAUDE.md' : null,
    ].filter(Boolean).join(', ')}`);
  }

  // Discover additional directories mounted at /workspace/extra/*
  // These are passed to the SDK so their CLAUDE.md files are loaded automatically
  const extraDirs: string[] = [];
  const extraBase = '/workspace/extra';
  if (fs.existsSync(extraBase)) {
    for (const entry of fs.readdirSync(extraBase)) {
      const fullPath = path.join(extraBase, entry);
      if (fs.statSync(fullPath).isDirectory()) {
        extraDirs.push(fullPath);
      }
    }
  }
  if (extraDirs.length > 0) {
    log(`Additional directories: ${extraDirs.join(', ')}`);
  }

  for await (const message of query({
    prompt: stream,
    options: {
      cwd: '/workspace/group',
      additionalDirectories: extraDirs.length > 0 ? extraDirs : undefined,
      resume: sessionId,
      resumeSessionAt: resumeAt,
      systemPrompt: systemPromptParts.length > 0
        ? {
          type: 'preset' as const,
          preset: 'claude_code' as const,
          append: systemPromptParts.join('\n\n'),
        }
        : undefined,
      allowedTools: [
        'Bash',
        'Read', 'Write', 'Edit', 'Glob', 'Grep',
        'WebSearch', 'WebFetch',
        'Task', 'TaskOutput', 'TaskStop',
        'TeamCreate', 'TeamDelete', 'SendMessage',
        'TodoWrite', 'ToolSearch', 'Skill',
        'NotebookEdit',
        'mcp__nanoclaw__*'
      ],
      env: sdkEnv,
      permissionMode: 'bypassPermissions',
      allowDangerouslySkipPermissions: true,
      settingSources: ['project', 'user'],
      mcpServers: {
        nanoclaw: {
          command: 'node',
          args: [mcpServerPath],
          env: {
            NANOCLAW_CHAT_JID: containerInput.chatJid,
            NANOCLAW_GROUP_FOLDER: containerInput.groupFolder,
            NANOCLAW_IS_MAIN: containerInput.isMain ? '1' : '0',
          },
        },
      },
      hooks: {
        PreCompact: [{ hooks: [createPreCompactHook(containerInput.assistantName)] }],
        PreToolUse: [{ matcher: 'Bash', hooks: [createSanitizeBashHook()] }],
      },
    }
  })) {
    messageCount++;
    const msgType = message.type === 'system' ? `system/${(message as { subtype?: string }).subtype}` : message.type;
    log(`[msg #${messageCount}] type=${msgType}`);

    if (message.type === 'assistant' && 'uuid' in message) {
      lastAssistantUuid = (message as { uuid: string }).uuid;
    }

    if (message.type === 'system' && message.subtype === 'init') {
      newSessionId = message.session_id;
      log(`Session initialized: ${newSessionId}`);
    }

    if (message.type === 'system' && (message as { subtype?: string }).subtype === 'task_notification') {
      const tn = message as { task_id: string; status: string; summary: string };
      log(`Task notification: task=${tn.task_id} status=${tn.status} summary=${tn.summary}`);
    }

    if (message.type === 'result') {
      resultCount++;
      const textResult = 'result' in message ? (message as { result?: string }).result : null;
      log(`Result #${resultCount}: subtype=${message.subtype}${textResult ? ` text=${textResult.slice(0, 200)}` : ''}`);
      writeOutput({
        status: 'success',
        result: textResult || null,
        newSessionId
      });
    }
  }

  ipcPolling = false;
  log(`Query done. Messages: ${messageCount}, results: ${resultCount}, lastAssistantUuid: ${lastAssistantUuid || 'none'}, closedDuringQuery: ${closedDuringQuery}`);
  return { newSessionId, lastAssistantUuid, closedDuringQuery };
}

function randomId(): string {
  return Math.random().toString(36).slice(2, 10);
}

function resolveProvider(
  sdkEnv: Record<string, string | undefined>,
): 'claude' | 'gemini' {
  const provider = (sdkEnv.LLM_PROVIDER || 'claude').toLowerCase();
  return provider === 'gemini' ? 'gemini' : 'claude';
}

function readGlobalMemory(containerInput: ContainerInput): string | undefined {
  const globalClaudeMdPath = '/workspace/global/CLAUDE.md';
  if (!containerInput.isMain && fs.existsSync(globalClaudeMdPath)) {
    return fs.readFileSync(globalClaudeMdPath, 'utf-8');
  }
  return undefined;
}

function readSoulMemory(): string | undefined {
  const soulMdPath = '/workspace/SOUL.md';
  if (fs.existsSync(soulMdPath)) {
    return fs.readFileSync(soulMdPath, 'utf-8');
  }
  return undefined;
}

function decodeXmlEntities(text: string): string {
  return text
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&amp;/g, '&');
}

function extractLatestMessageText(prompt: string): string {
  const regex = /<message\b[^>]*>([\s\S]*?)<\/message>/g;
  let match: RegExpExecArray | null;
  let latest = '';
  while ((match = regex.exec(prompt)) !== null) {
    latest = match[1] || '';
  }
  return decodeXmlEntities(latest).trim() || prompt.trim();
}

function shouldUseWebSearch(text: string): boolean {
  return /\b(news|latest|today|now|current|update|updates|headlines|breaking|source|sources|link|links|youtube)\b/i.test(
    text,
  );
}

function stripHtmlTags(html: string): string {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

function decodeDdgUrl(rawHref: string): string {
  try {
    const abs = new URL(rawHref, 'https://duckduckgo.com');
    const uddg = abs.searchParams.get('uddg');
    if (uddg) return decodeURIComponent(uddg);
    return abs.toString();
  } catch {
    return rawHref;
  }
}

interface WebResult {
  title: string;
  url: string;
  snippet: string;
}

async function searchWithSerper(
  queryText: string,
  apiKey: string,
  limit: number,
): Promise<WebResult[]> {
  const resp = await fetchWithTimeout(
    'https://google.serper.dev/search',
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-KEY': apiKey,
      },
      body: JSON.stringify({ q: queryText, num: limit }),
    },
    12_000,
  );
  if (!resp.ok) {
    throw new Error(`Serper search failed (${resp.status})`);
  }
  const data = (await resp.json()) as {
    organic?: Array<{ title?: string; link?: string; snippet?: string }>;
  };
  return (data.organic || [])
    .map((r) => ({
      title: (r.title || '').trim(),
      url: (r.link || '').trim(),
      snippet: (r.snippet || '').trim(),
    }))
    .filter((r) => r.title && r.url)
    .slice(0, limit);
}

async function searchWithDuckDuckGo(
  queryText: string,
  limit: number,
): Promise<WebResult[]> {
  const url = `https://duckduckgo.com/html/?q=${encodeURIComponent(queryText)}`;
  const resp = await fetchWithTimeout(url, { method: 'GET' }, 12_000);
  if (!resp.ok) {
    throw new Error(`DuckDuckGo search failed (${resp.status})`);
  }
  const html = await resp.text();
  const out: WebResult[] = [];
  const anchorRe =
    /<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi;
  let m: RegExpExecArray | null;
  while ((m = anchorRe.exec(html)) !== null && out.length < limit) {
    const rawHref = m[1] || '';
    const titleRaw = m[2] || '';
    const title = stripHtmlTags(decodeXmlEntities(titleRaw));
    const link = decodeDdgUrl(decodeXmlEntities(rawHref));
    if (!title || !/^https?:\/\//i.test(link)) continue;

    // Try to read nearby snippet block
    const tail = html.slice(m.index, Math.min(html.length, m.index + 1600));
    const snippetMatch =
      /class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)<\/a>/i.exec(tail) ||
      /class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)<\/div>/i.exec(tail);
    const snippet = snippetMatch
      ? stripHtmlTags(decodeXmlEntities(snippetMatch[1] || ''))
      : '';
    out.push({ title, url: link, snippet });
  }
  return out;
}

async function fetchPageSnippet(url: string): Promise<string> {
  try {
    const resp = await fetchWithTimeout(
      url,
      {
        method: 'GET',
        headers: {
          'User-Agent': 'Mozilla/5.0 (compatible; NanoClaw/1.0)',
        },
      },
      8_000,
    );
    if (!resp.ok) return '';
    const html = await resp.text();
    const titleMatch = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(html);
    const descMatch =
      /<meta[^>]+name=["']description["'][^>]+content=["']([^"']+)["'][^>]*>/i.exec(
        html,
      ) ||
      /<meta[^>]+content=["']([^"']+)["'][^>]+name=["']description["'][^>]*>/i.exec(
        html,
      );
    const title = titleMatch ? stripHtmlTags(decodeXmlEntities(titleMatch[1])) : '';
    const desc = descMatch ? stripHtmlTags(decodeXmlEntities(descMatch[1])) : '';
    return [title, desc].filter(Boolean).join(' - ').slice(0, 300);
  } catch {
    return '';
  }
}

async function buildWebContext(
  latestUserMessage: string,
  sdkEnv: Record<string, string | undefined>,
): Promise<string> {
  if (!shouldUseWebSearch(latestUserMessage)) return '';

  const limit = parseInt(sdkEnv.WEB_SEARCH_RESULTS || `${DEFAULT_WEB_RESULTS}`, 10) || DEFAULT_WEB_RESULTS;
  const serperKey = sdkEnv.SERPER_API_KEY;
  let results: WebResult[] = [];
  try {
    results = serperKey
      ? await searchWithSerper(latestUserMessage, serperKey, limit)
      : await searchWithDuckDuckGo(latestUserMessage, limit);
  } catch (err) {
    return `WEB_SEARCH_ERROR: ${err instanceof Error ? err.message : String(err)}`;
  }

  if (/youtube/i.test(latestUserMessage)) {
    const ytFirst = results.filter((r) => /youtube\.com|youtu\.be/i.test(r.url));
    const rest = results.filter((r) => !/youtube\.com|youtu\.be/i.test(r.url));
    results = [...ytFirst, ...rest];
  }

  const enriched = await Promise.all(
    results.slice(0, limit).map(async (r) => {
      const pageSnippet = await fetchPageSnippet(r.url);
      return {
        ...r,
        pageSnippet,
      };
    }),
  );

  if (enriched.length === 0) return 'WEB_SEARCH_RESULTS: none';
  const lines: string[] = [];
  lines.push(`WEB_SEARCH_RESULTS (${new Date().toISOString()}):`);
  enriched.forEach((r, idx) => {
    lines.push(`[${idx + 1}] ${r.title}`);
    lines.push(`URL: ${r.url}`);
    if (r.snippet) lines.push(`Snippet: ${r.snippet}`);
    if (r.pageSnippet) lines.push(`Page: ${r.pageSnippet}`);
  });
  return lines.join('\n');
}

async function runGeminiTurn(
  prompt: string,
  sessionId: string,
  containerInput: ContainerInput,
  sdkEnv: Record<string, string | undefined>,
): Promise<void> {
  const apiKey = sdkEnv.GEMINI_API_KEY || sdkEnv.GOOGLE_API_KEY;
  if (!apiKey) {
    throw new Error('Missing GEMINI_API_KEY (or GOOGLE_API_KEY) for Gemini provider');
  }

  const model = sdkEnv.LLM_MODEL || DEFAULT_GEMINI_MODEL;
  const history = geminiHistoryBySession.get(sessionId) || [];
  const soulPrompt = readSoulMemory();
  const globalPrompt = readGlobalMemory(containerInput);
  const systemPromptParts = [soulPrompt, globalPrompt].filter(
    (part): part is string => Boolean(part),
  );
  const systemPrompt = systemPromptParts.length > 0
    ? systemPromptParts.join('\n\n')
    : undefined;
  const latestUserMessage = extractLatestMessageText(prompt);
  const webContext = await buildWebContext(latestUserMessage, sdkEnv);

  if (systemPromptParts.length > 0) {
    log(`Gemini system context loaded from: ${[
      soulPrompt ? 'SOUL.md' : null,
      globalPrompt ? 'global/CLAUDE.md' : null,
    ].filter(Boolean).join(', ')}`);
  }

  const behavioralInstruction =
    'You are Andy, an assistant in chat. Reply directly to the user\'s latest message. ' +
    'Do not describe, summarize, or classify what the user said. ' +
    'Use prior messages only as context. Keep responses concise and actionable unless the user asks for detail. ' +
    'If WEB_SEARCH_RESULTS are provided, prioritize them for factual/current answers and include source URLs in your response.';

  const currentTurnText = webContext
    ? `${latestUserMessage}\n\n${webContext}`
    : latestUserMessage;

  const payload: Record<string, unknown> = {
    contents: [
      ...history.map((m) => ({
        role: m.role,
        parts: [{ text: m.text }],
      })),
      {
        role: 'user',
        parts: [{ text: currentTurnText }],
      },
    ],
  };

  if (systemPrompt) {
    payload.systemInstruction = {
      parts: [{ text: `${behavioralInstruction}\n\n${systemPrompt}` }],
    };
  } else {
    payload.systemInstruction = {
      parts: [{ text: behavioralInstruction }],
    };
  }

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(apiKey)}`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  const data = (await response.json()) as {
    error?: { message?: string };
    candidates?: Array<{
      finishReason?: string;
      content?: { parts?: Array<{ text?: string }> };
    }>;
  };

  if (!response.ok || data.error) {
    const errMessage =
      data.error?.message || `Gemini API error (status ${response.status})`;
    throw new Error(errMessage);
  }

  const text = (data.candidates?.[0]?.content?.parts || [])
    .map((p) => p.text || '')
    .join('')
    .trim();

  if (!text) {
    const reason = data.candidates?.[0]?.finishReason || 'empty response';
    throw new Error(`Gemini returned no text (${reason})`);
  }

  history.push({ role: 'user', text: latestUserMessage });
  history.push({ role: 'model', text });
  geminiHistoryBySession.set(sessionId, history);

  writeOutput({
    status: 'success',
    result: text,
    newSessionId: sessionId,
  });
}

async function main(): Promise<void> {
  let containerInput: ContainerInput;

  try {
    const stdinData = await readStdin();
    containerInput = JSON.parse(stdinData);
    // Delete the temp file the entrypoint wrote — it contains secrets
    try { fs.unlinkSync('/tmp/input.json'); } catch { /* may not exist */ }
    log(`Received input for group: ${containerInput.groupFolder}`);
  } catch (err) {
    writeOutput({
      status: 'error',
      result: null,
      error: `Failed to parse input: ${err instanceof Error ? err.message : String(err)}`
    });
    process.exit(1);
  }

  // Build SDK env: merge secrets into process.env for the SDK only.
  // Secrets never touch process.env itself, so Bash subprocesses can't see them.
  const sdkEnv: Record<string, string | undefined> = { ...process.env };
  for (const [key, value] of Object.entries(containerInput.secrets || {})) {
    sdkEnv[key] = value;
  }

  const provider = resolveProvider(sdkEnv);
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const mcpServerPath = path.join(__dirname, 'ipc-mcp-stdio.js');

  let sessionId = containerInput.sessionId;
  fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });

  // Clean up stale _close sentinel from previous container runs
  try { fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL); } catch { /* ignore */ }

  // Build initial prompt (drain any pending IPC messages too)
  let prompt = containerInput.prompt;
  if (containerInput.isScheduledTask) {
    prompt = `[SCHEDULED TASK - The following message was sent automatically and is not coming directly from the user or group.]\n\n${prompt}`;
  }
  const pending = drainIpcInput();
  if (pending.length > 0) {
    log(`Draining ${pending.length} pending IPC messages into initial prompt`);
    prompt += '\n' + pending.join('\n');
  }

  if (provider === 'gemini' && !sessionId) {
    sessionId = `gemini-${Date.now()}-${randomId()}`;
  }

  // Query loop: run query → wait for IPC message → run new query → repeat
  let resumeAt: string | undefined;
  try {
    while (true) {
      log(`Starting ${provider} query (session: ${sessionId || 'new'}, resumeAt: ${resumeAt || 'latest'})...`);

      if (provider === 'gemini') {
        await runGeminiTurn(prompt, sessionId!, containerInput, sdkEnv);
      } else {
        const queryResult = await runQuery(prompt, sessionId, mcpServerPath, containerInput, sdkEnv, resumeAt);
        if (queryResult.newSessionId) {
          sessionId = queryResult.newSessionId;
        }
        if (queryResult.lastAssistantUuid) {
          resumeAt = queryResult.lastAssistantUuid;
        }

        // If _close was consumed during the query, exit immediately.
        // Don't emit a session-update marker (it would reset the host's
        // idle timer and cause a 30-min delay before the next _close).
        if (queryResult.closedDuringQuery) {
          log('Close sentinel consumed during query, exiting');
          break;
        }
      }

      // Emit session update so host can track it
      writeOutput({ status: 'success', result: null, newSessionId: sessionId });

      log('Query ended, waiting for next IPC message...');

      // Wait for the next message or _close sentinel
      const nextMessage = await waitForIpcMessage();
      if (nextMessage === null) {
        log('Close sentinel received, exiting');
        break;
      }

      log(`Got new message (${nextMessage.length} chars), starting new query`);
      prompt = nextMessage;
    }
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    log(`Agent error: ${errorMessage}`);
    writeOutput({
      status: 'error',
      result: null,
      newSessionId: sessionId,
      error: errorMessage
    });
    process.exit(1);
  }
}

main();
