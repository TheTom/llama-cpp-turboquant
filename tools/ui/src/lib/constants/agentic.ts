import type { AgenticConfig } from '$lib/types/agentic';

export const ATTACHMENT_SAVED_REGEX = /\[Attachment saved: ([^\]]+)\]/;

export const NEWLINE_SEPARATOR = '\n';

export const DEFAULT_AGENTIC_CONFIG: AgenticConfig = {
	enabled: true,
	maxTurns: 100,
	maxToolPreviewLines: 25
} as const;

/** Injected when agentic tools are enabled so models use tools more reliably. */
export const CODING_AGENT_SYSTEM_PROMPT = [
	'You are a coding agent with access to tools.',
	'Prefer specialized tools over shell when possible: read_file/file_glob_search/grep_search for files, git_* for repository inspection, web_search then fetch_url for online research, run_python for Python, and exec_shell_command for bash/PowerShell/cmd.',
	'Investigate before editing. Keep tool calls focused and explain results briefly after tools return.',
	'Do not invent file contents, URLs, or Git history; use tools instead.'
].join(' ');

export const CODING_AGENT_SYSTEM_PROMPT_MARKER = 'You are a coding agent with access to tools.';

export const REASONING_TAGS = {
	START: '<think>',
	END: '</think>'
} as const;

/**
 * @deprecated Legacy marker tags - only used for migration of old stored messages.
 * New messages use structured fields (reasoningContent, toolCalls, toolCallId).
 */
export const LEGACY_AGENTIC_TAGS = {
	TOOL_CALL_START: '<<<AGENTIC_TOOL_CALL_START>>>',
	TOOL_CALL_END: '<<<AGENTIC_TOOL_CALL_END>>>',
	TOOL_NAME_PREFIX: '<<<TOOL_NAME:',
	TOOL_ARGS_START: '<<<TOOL_ARGS_START>>>',
	TOOL_ARGS_END: '<<<TOOL_ARGS_END>>>',
	TAG_SUFFIX: '>>>'
} as const;

/**
 * @deprecated Legacy reasoning tags - only used for migration of old stored messages.
 * New messages use the dedicated reasoningContent field.
 */
export const LEGACY_REASONING_TAGS = {
	START: '<<<reasoning_content_start>>>',
	END: '<<<reasoning_content_end>>>'
} as const;

/**
 * @deprecated Legacy regex patterns - only used for migration of old stored messages.
 */
export const LEGACY_AGENTIC_REGEX = {
	COMPLETED_TOOL_CALL:
		/<<<AGENTIC_TOOL_CALL_START>>>\n<<<TOOL_NAME:(.+?)>>>\n<<<TOOL_ARGS_START>>>([\s\S]*?)<<<TOOL_ARGS_END>>>([\s\S]*?)<<<AGENTIC_TOOL_CALL_END>>>/g,
	REASONING_BLOCK: /<<<reasoning_content_start>>>[\s\S]*?<<<reasoning_content_end>>>/g,
	REASONING_EXTRACT: /<<<reasoning_content_start>>>([\s\S]*?)<<<reasoning_content_end>>>/,
	REASONING_OPEN: /<<<reasoning_content_start>>>[\s\S]*$/,
	AGENTIC_TOOL_CALL_BLOCK: /\n*<<<AGENTIC_TOOL_CALL_START>>>[\s\S]*?<<<AGENTIC_TOOL_CALL_END>>>/g,
	AGENTIC_TOOL_CALL_OPEN: /\n*<<<AGENTIC_TOOL_CALL_START>>>[\s\S]*$/,
	HAS_LEGACY_MARKERS: /<<<(?:AGENTIC_TOOL_CALL_START|reasoning_content_start)>>>/
} as const;
