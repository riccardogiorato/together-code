/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
    CountTokensResponse,
    GenerateContentResponse,
    GenerateContentParameters,
    CountTokensParameters,
    EmbedContentResponse,
    EmbedContentParameters,
    FinishReason,
    Part,
    Content,
    Tool,
    ToolListUnion,
    CallableTool,
    FunctionCall,
    FunctionResponse,
} from '@google/genai';
import { ContentGenerator } from './contentGenerator.js';
import Together from 'together-ai';
import { logApiResponse } from '../telemetry/loggers.js';
import { ApiResponseEvent } from '../telemetry/types.js';
import { Config } from '../config/config.js';
import { openaiLogger } from '../utils/openaiLogger.js';
import { UserTierId } from '../code_assist/types.js';

// Together AI API type definitions for logging
interface TogetherToolCall {
    id: string;
    type: 'function';
    function: {
        name: string;
        arguments: string;
    };
}

interface TogetherMessage {
    role: 'system' | 'user' | 'assistant' | 'tool';
    content: string | null;
    tool_calls?: TogetherToolCall[];
    tool_call_id?: string;
}

interface TogetherUsage {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
}

interface TogetherChoice {
    index: number;
    message: TogetherMessage;
    finish_reason: string;
}

interface TogetherRequestFormat {
    model: string;
    messages: TogetherMessage[];
    temperature?: number;
    max_tokens?: number;
    top_p?: number;
    tools?: unknown[];
}

interface TogetherResponseFormat {
    id: string;
    object: string;
    created: number;
    model: string;
    choices: TogetherChoice[];
    usage?: TogetherUsage;
}

export class TogetherContentGenerator implements ContentGenerator {
    private client: Together;
    private model: string;
    private config: Config;
    private streamingToolCalls: Map<
        number,
        {
            id?: string;
            name?: string;
            arguments: string;
        }
    > = new Map();

    constructor(apiKey: string, model: string, config: Config) {
        this.model = model;
        this.config = config;

        // Configure timeout settings
        const timeoutConfig = {
            timeout: 120000, // 2 minutes default
            maxRetries: 3,
        };

        // Allow config to override timeout settings
        const contentGeneratorConfig = this.config.getContentGeneratorConfig();
        if (contentGeneratorConfig?.timeout) {
            timeoutConfig.timeout = contentGeneratorConfig.timeout;
        }
        if (contentGeneratorConfig?.maxRetries !== undefined) {
            timeoutConfig.maxRetries = contentGeneratorConfig.maxRetries;
        }

        this.client = new Together({
            apiKey,
            timeout: timeoutConfig.timeout,
            maxRetries: timeoutConfig.maxRetries,
        });
    }

    /**
     * Check if an error is a timeout error
     */
    private isTimeoutError(error: unknown): boolean {
        if (!error) return false;

        const errorMessage =
            error instanceof Error
                ? error.message.toLowerCase()
                : String(error).toLowerCase();
        const errorCode = (error as any)?.code;
        const errorType = (error as any)?.type;

        return (
            errorMessage.includes('timeout') ||
            errorMessage.includes('timed out') ||
            errorMessage.includes('connection timeout') ||
            errorMessage.includes('request timeout') ||
            errorMessage.includes('read timeout') ||
            errorMessage.includes('etimedout') ||
            errorMessage.includes('esockettimedout') ||
            errorCode === 'ETIMEDOUT' ||
            errorCode === 'ESOCKETTIMEDOUT' ||
            errorType === 'timeout' ||
            errorMessage.includes('request timed out') ||
            errorMessage.includes('deadline exceeded')
        );
    }

    async generateContent(
        request: GenerateContentParameters,
    ): Promise<GenerateContentResponse> {
        const startTime = Date.now();
        const messages = this.convertToTogetherFormat(request);

        try {
            const samplingParams = this.buildSamplingParameters(request);

            const createParams: any = {
                model: this.model,
                messages,
                ...samplingParams,
            };

            if (request.config?.tools) {
                createParams.tools = await this.convertGeminiToolsToTogether(
                    request.config.tools,
                );
            }

            const completion = await this.client.chat.completions.create(createParams);
            const response = this.convertToGeminiFormat(completion);
            const durationMs = Date.now() - startTime;

            // Log API response event for UI telemetry
            const responseEvent = new ApiResponseEvent(
                this.model,
                durationMs,
                `together-${Date.now()}`,
                this.config.getContentGeneratorConfig()?.authType,
                response.usageMetadata,
            );

            logApiResponse(this.config, responseEvent);

            // Log interaction if enabled
            if (this.config.getContentGeneratorConfig()?.enableOpenAILogging) {
                const togetherRequest = await this.convertGeminiRequestToTogether(request);
                const togetherResponse = this.convertGeminiResponseToTogether(response);
                await openaiLogger.logInteraction(togetherRequest, togetherResponse);
            }

            return response;
        } catch (error) {
            const durationMs = Date.now() - startTime;
            const isTimeoutError = this.isTimeoutError(error);
            const errorMessage = isTimeoutError
                ? `Request timeout after ${Math.round(durationMs / 1000)}s. Try reducing input length or increasing timeout in config.`
                : error instanceof Error
                    ? error.message
                    : String(error);

            // Estimate token usage for failed requests
            let estimatedUsage;
            try {
                const tokenCountResult = await this.countTokens({
                    contents: request.contents,
                    model: this.model,
                });
                estimatedUsage = {
                    promptTokenCount: tokenCountResult.totalTokens,
                    candidatesTokenCount: 0,
                    totalTokenCount: tokenCountResult.totalTokens,
                };
            } catch {
                const contentStr = JSON.stringify(request.contents);
                const estimatedTokens = Math.ceil(contentStr.length / 4);
                estimatedUsage = {
                    promptTokenCount: estimatedTokens,
                    candidatesTokenCount: 0,
                    totalTokenCount: estimatedTokens,
                };
            }

            // Log API error event
            const errorEvent = new ApiResponseEvent(
                this.model,
                durationMs,
                `together-${Date.now()}`,
                this.config.getContentGeneratorConfig()?.authType,
                estimatedUsage,
                undefined,
                errorMessage,
            );
            logApiResponse(this.config, errorEvent);

            // Log error interaction if enabled
            if (this.config.getContentGeneratorConfig()?.enableOpenAILogging) {
                const togetherRequest = await this.convertGeminiRequestToTogether(request);
                await openaiLogger.logInteraction(
                    togetherRequest,
                    undefined,
                    error as Error,
                );
            }

            console.error('Together AI API Error:', errorMessage);

            if (isTimeoutError) {
                throw new Error(
                    `${errorMessage}\n\nTroubleshooting tips:\n` +
                    `- Reduce input length or complexity\n` +
                    `- Increase timeout in config: contentGenerator.timeout\n` +
                    `- Check network connectivity\n` +
                    `- Consider using streaming mode for long responses`,
                );
            }

            throw new Error(`Together AI API error: ${errorMessage}`);
        }
    }

    async generateContentStream(
        request: GenerateContentParameters,
    ): Promise<AsyncGenerator<GenerateContentResponse>> {
        const startTime = Date.now();
        const messages = this.convertToTogetherFormat(request);

        try {
            const samplingParams = this.buildSamplingParameters(request);

            const createParams: any = {
                model: this.model,
                messages,
                ...samplingParams,
                stream: true,
            };

            if (request.config?.tools) {
                createParams.tools = await this.convertGeminiToolsToTogether(
                    request.config.tools,
                );
            }

            const stream = await this.client.chat.completions.create(createParams) as unknown as AsyncIterable<any>;
            const originalStream = this.streamGenerator(stream);

            // Collect responses for final logging
            const responses: GenerateContentResponse[] = [];

            const wrappedGenerator = async function* (this: TogetherContentGenerator) {
                try {
                    for await (const response of originalStream) {
                        responses.push(response);
                        yield response;
                    }

                    const durationMs = Date.now() - startTime;
                    const finalUsageMetadata = responses
                        .slice()
                        .reverse()
                        .find((r) => r.usageMetadata)?.usageMetadata;

                    // Log API response event
                    const responseEvent = new ApiResponseEvent(
                        this.model,
                        durationMs,
                        `together-stream-${Date.now()}`,
                        this.config.getContentGeneratorConfig()?.authType,
                        finalUsageMetadata,
                    );

                    logApiResponse(this.config, responseEvent);

                    // Log interaction if enabled
                    if (this.config.getContentGeneratorConfig()?.enableOpenAILogging) {
                        const togetherRequest = await this.convertGeminiRequestToTogether(request);
                        const combinedResponse = this.combineStreamResponsesForLogging(responses);
                        const togetherResponse = this.convertGeminiResponseToTogether(combinedResponse);
                        await openaiLogger.logInteraction(togetherRequest, togetherResponse);
                    }
                } catch (error) {
                    const durationMs = Date.now() - startTime;
                    const isTimeoutError = this.isTimeoutError(error);
                    const errorMessage = isTimeoutError
                        ? `Streaming request timeout after ${Math.round(durationMs / 1000)}s.`
                        : error instanceof Error
                            ? error.message
                            : String(error);

                    // Handle streaming errors similar to generateContent
                    let estimatedUsage;
                    try {
                        const tokenCountResult = await this.countTokens({
                            contents: request.contents,
                            model: this.model,
                        });
                        estimatedUsage = {
                            promptTokenCount: tokenCountResult.totalTokens,
                            candidatesTokenCount: 0,
                            totalTokenCount: tokenCountResult.totalTokens,
                        };
                    } catch {
                        const contentStr = JSON.stringify(request.contents);
                        const estimatedTokens = Math.ceil(contentStr.length / 4);
                        estimatedUsage = {
                            promptTokenCount: estimatedTokens,
                            candidatesTokenCount: 0,
                            totalTokenCount: estimatedTokens,
                        };
                    }

                    const errorEvent = new ApiResponseEvent(
                        this.model,
                        durationMs,
                        `together-stream-${Date.now()}`,
                        this.config.getContentGeneratorConfig()?.authType,
                        estimatedUsage,
                        undefined,
                        errorMessage,
                    );
                    logApiResponse(this.config, errorEvent);

                    if (this.config.getContentGeneratorConfig()?.enableOpenAILogging) {
                        const togetherRequest = await this.convertGeminiRequestToTogether(request);
                        await openaiLogger.logInteraction(
                            togetherRequest,
                            undefined,
                            error as Error,
                        );
                    }

                    if (isTimeoutError) {
                        throw new Error(
                            `${errorMessage}\n\nStreaming timeout troubleshooting:\n` +
                            `- Reduce input length or complexity\n` +
                            `- Increase timeout in config: contentGenerator.timeout\n` +
                            `- Check network stability for streaming connections`,
                        );
                    }

                    throw error;
                }
            }.bind(this);

            return wrappedGenerator();
        } catch (error) {
            const durationMs = Date.now() - startTime;
            const isTimeoutError = this.isTimeoutError(error);
            const errorMessage = isTimeoutError
                ? `Streaming setup timeout after ${Math.round(durationMs / 1000)}s.`
                : error instanceof Error
                    ? error.message
                    : String(error);

            console.error('Together AI Streaming Error:', errorMessage);

            if (isTimeoutError) {
                throw new Error(
                    `${errorMessage}\n\nStreaming setup troubleshooting:\n` +
                    `- Reduce input length or complexity\n` +
                    `- Increase timeout in config: contentGenerator.timeout\n` +
                    `- Check network connectivity and firewall settings`,
                );
            }

            throw new Error(`Together AI API error: ${errorMessage}`);
        }
    }

    private async *streamGenerator(
        stream: AsyncIterable<any>,
    ): AsyncGenerator<GenerateContentResponse> {
        this.streamingToolCalls.clear();

        for await (const chunk of stream) {
            yield this.convertStreamChunkToGeminiFormat(chunk);
        }
    }

    private combineStreamResponsesForLogging(
        responses: GenerateContentResponse[],
    ): GenerateContentResponse {
        if (responses.length === 0) {
            return new GenerateContentResponse();
        }

        const finalUsageMetadata = responses
            .slice()
            .reverse()
            .find((r) => r.usageMetadata)?.usageMetadata;

        const combinedParts: Part[] = [];
        let combinedText = '';
        const functionCalls: Part[] = [];

        for (const response of responses) {
            if (response.candidates?.[0]?.content?.parts) {
                for (const part of response.candidates[0].content.parts) {
                    if ('text' in part && part.text) {
                        combinedText += part.text;
                    } else if ('functionCall' in part && part.functionCall) {
                        functionCalls.push(part);
                    }
                }
            }
        }

        if (combinedText) {
            combinedParts.push({ text: combinedText });
        }
        combinedParts.push(...functionCalls);

        const combinedResponse = new GenerateContentResponse();
        combinedResponse.candidates = [
            {
                content: {
                    parts: combinedParts,
                    role: 'model' as const,
                },
                finishReason:
                    responses[responses.length - 1]?.candidates?.[0]?.finishReason ||
                    FinishReason.FINISH_REASON_UNSPECIFIED,
                index: 0,
                safetyRatings: [],
            },
        ];
        combinedResponse.modelVersion = this.model;
        combinedResponse.promptFeedback = { safetyRatings: [] };
        combinedResponse.usageMetadata = finalUsageMetadata;

        return combinedResponse;
    }

    async countTokens(
        request: CountTokensParameters,
    ): Promise<CountTokensResponse> {
        // Together AI doesn't have a direct token counting endpoint
        // Use rough estimation: 1 token ≈ 4 characters
        const content = JSON.stringify(request.contents);
        const estimatedTokens = Math.ceil(content.length / 4);

        return {
            totalTokens: estimatedTokens,
        };
    }

    async embedContent(
        request: EmbedContentParameters,
    ): Promise<EmbedContentResponse> {
        // Extract text from contents
        let text = '';
        if (Array.isArray(request.contents)) {
            text = request.contents
                .map((content) => {
                    if (typeof content === 'string') return content;
                    if ('parts' in content && content.parts) {
                        return content.parts
                            .map((part) =>
                                typeof part === 'string'
                                    ? part
                                    : 'text' in part
                                        ? (part as { text?: string }).text || ''
                                        : '',
                            )
                            .join(' ');
                    }
                    return '';
                })
                .join(' ');
        } else if (request.contents) {
            if (typeof request.contents === 'string') {
                text = request.contents;
            } else if ('parts' in request.contents && request.contents.parts) {
                text = request.contents.parts
                    .map((part: Part) =>
                        typeof part === 'string' ? part : 'text' in part ? part.text : '',
                    )
                    .join(' ');
            }
        }

        try {
            const embedding = await this.client.embeddings.create({
                model: 'togethercomputer/m2-bert-80M-8k-retrieval', // Together AI embedding model
                input: text,
            });

            return {
                embeddings: [
                    {
                        values: embedding.data[0].embedding,
                    },
                ],
            };
        } catch (error) {
            console.error('Together AI Embedding Error:', error);
            throw new Error(
                `Together AI API error: ${error instanceof Error ? error.message : String(error)}`,
            );
        }
    }

    async getTier(): Promise<UserTierId | undefined> {
        // Together AI doesn't have tier information like Google's services
        // Return undefined to indicate no tier restrictions
        return undefined;
    }

    private buildSamplingParameters(request: GenerateContentParameters): any {
        const params: any = {};

        // Get config-level sampling parameters
        const configSamplingParams = this.config.getContentGeneratorConfig()?.samplingParams;

        // Apply config-level parameters first
        if (configSamplingParams) {
            if (configSamplingParams.temperature !== undefined) {
                params.temperature = configSamplingParams.temperature;
            }
            if (configSamplingParams.top_p !== undefined) {
                params.top_p = configSamplingParams.top_p;
            }
            if (configSamplingParams.max_tokens !== undefined) {
                params.max_tokens = configSamplingParams.max_tokens;
            }
            if (configSamplingParams.frequency_penalty !== undefined) {
                params.frequency_penalty = configSamplingParams.frequency_penalty;
            }
            if (configSamplingParams.presence_penalty !== undefined) {
                params.presence_penalty = configSamplingParams.presence_penalty;
            }
        }

        // Apply request-level parameters (highest priority)
        if (request.config?.temperature !== undefined) {
            params.temperature = request.config.temperature;
        }
        if (request.config?.topP !== undefined) {
            params.top_p = request.config.topP;
        }
        if (request.config?.maxOutputTokens !== undefined) {
            params.max_tokens = request.config.maxOutputTokens;
        }

        return params;
    }

    private convertGeminiParametersToTogether(
        parameters: Record<string, unknown>,
    ): Record<string, unknown> | undefined {
        if (!parameters || typeof parameters !== 'object') {
            return parameters;
        }

        const converted = JSON.parse(JSON.stringify(parameters));

        const convertTypes = (obj: unknown): unknown => {
            if (typeof obj !== 'object' || obj === null) {
                return obj;
            }

            if (Array.isArray(obj)) {
                return obj.map(convertTypes);
            }

            const result: Record<string, unknown> = {};
            for (const [key, value] of Object.entries(obj)) {
                if (key === 'type' && typeof value === 'string') {
                    const lowerValue = value.toLowerCase();
                    if (lowerValue === 'integer') {
                        result[key] = 'integer';
                    } else if (lowerValue === 'number') {
                        result[key] = 'number';
                    } else {
                        result[key] = lowerValue;
                    }
                } else if (
                    key === 'minimum' ||
                    key === 'maximum' ||
                    key === 'multipleOf'
                ) {
                    if (typeof value === 'string' && !isNaN(Number(value))) {
                        result[key] = Number(value);
                    } else {
                        result[key] = value;
                    }
                } else if (
                    key === 'minLength' ||
                    key === 'maxLength' ||
                    key === 'minItems' ||
                    key === 'maxItems'
                ) {
                    if (typeof value === 'string' && !isNaN(Number(value))) {
                        result[key] = parseInt(value, 10);
                    } else {
                        result[key] = value;
                    }
                } else if (typeof value === 'object') {
                    result[key] = convertTypes(value);
                } else {
                    result[key] = value;
                }
            }
            return result;
        };

        return convertTypes(converted) as Record<string, unknown> | undefined;
    }

    private async convertGeminiToolsToTogether(
        geminiTools: ToolListUnion,
    ): Promise<any[]> {
        const togetherTools: any[] = [];

        for (const tool of geminiTools) {
            let actualTool: Tool;

            if ('tool' in tool) {
                actualTool = await (tool as CallableTool).tool();
            } else {
                actualTool = tool as Tool;
            }

            if (actualTool.functionDeclarations) {
                for (const func of actualTool.functionDeclarations) {
                    if (func.name && func.description) {
                        togetherTools.push({
                            type: 'function',
                            function: {
                                name: func.name,
                                description: func.description,
                                parameters: this.convertGeminiParametersToTogether(
                                    (func.parameters || {}) as Record<string, unknown>,
                                ),
                            },
                        });
                    }
                }
            }
        }

        return togetherTools;
    }

    private convertToTogetherFormat(
        request: GenerateContentParameters,
    ): any[] {
        const messages: any[] = [];

        // Handle system instruction
        if (request.config?.systemInstruction) {
            const systemInstruction = request.config.systemInstruction;
            let systemText = '';

            if (Array.isArray(systemInstruction)) {
                systemText = systemInstruction
                    .map((content) => {
                        if (typeof content === 'string') return content;
                        if ('parts' in content) {
                            const contentObj = content as Content;
                            return (
                                contentObj.parts
                                    ?.map((p: Part) =>
                                        typeof p === 'string' ? p : 'text' in p ? p.text : '',
                                    )
                                    .join('\n') || ''
                            );
                        }
                        return '';
                    })
                    .join('\n');
            } else if (typeof systemInstruction === 'string') {
                systemText = systemInstruction;
            } else if (
                typeof systemInstruction === 'object' &&
                'parts' in systemInstruction
            ) {
                const systemContent = systemInstruction as Content;
                systemText =
                    systemContent.parts
                        ?.map((p: Part) =>
                            typeof p === 'string' ? p : 'text' in p ? p.text : '',
                        )
                        .join('\n') || '';
            }

            if (systemText) {
                messages.push({
                    role: 'system',
                    content: systemText,
                });
            }
        }

        // Handle contents
        if (Array.isArray(request.contents)) {
            for (const content of request.contents) {
                if (typeof content === 'string') {
                    messages.push({ role: 'user', content });
                } else if ('role' in content && 'parts' in content) {
                    const functionCalls: FunctionCall[] = [];
                    const functionResponses: FunctionResponse[] = [];
                    const textParts: string[] = [];

                    for (const part of content.parts || []) {
                        if (typeof part === 'string') {
                            textParts.push(part);
                        } else if ('text' in part && part.text) {
                            textParts.push(part.text);
                        } else if ('functionCall' in part && part.functionCall) {
                            functionCalls.push(part.functionCall);
                        } else if ('functionResponse' in part && part.functionResponse) {
                            functionResponses.push(part.functionResponse);
                        }
                    }

                    // Handle function responses (tool results)
                    if (functionResponses.length > 0) {
                        for (const funcResponse of functionResponses) {
                            messages.push({
                                role: 'tool',
                                tool_call_id: funcResponse.id || '',
                                content:
                                    typeof funcResponse.response === 'string'
                                        ? funcResponse.response
                                        : JSON.stringify(funcResponse.response),
                            });
                        }
                    }
                    // Handle model messages with function calls
                    else if (content.role === 'model' && functionCalls.length > 0) {
                        const toolCalls = functionCalls.map((fc, index) => ({
                            id: fc.id || `call_${index}`,
                            type: 'function',
                            function: {
                                name: fc.name || '',
                                arguments: JSON.stringify(fc.args || {}),
                            },
                        }));

                        messages.push({
                            role: 'assistant',
                            content: textParts.join('\n') || null,
                            tool_calls: toolCalls,
                        });
                    }
                    // Handle regular text messages
                    else {
                        const role = content.role === 'model' ? 'assistant' : 'user';
                        const text = textParts.join('\n');
                        if (text) {
                            messages.push({ role, content: text });
                        }
                    }
                }
            }
        } else if (request.contents) {
            if (typeof request.contents === 'string') {
                messages.push({ role: 'user', content: request.contents });
            } else if ('role' in request.contents && 'parts' in request.contents) {
                const content = request.contents;
                const role = content.role === 'model' ? 'assistant' : 'user';
                const text =
                    content.parts
                        ?.map((p: Part) =>
                            typeof p === 'string' ? p : 'text' in p ? p.text : '',
                        )
                        .join('\n') || '';
                messages.push({ role, content: text });
            }
        }

        return this.cleanOrphanedToolCalls(messages);
    }

    private cleanOrphanedToolCalls(messages: any[]): any[] {
        const cleaned: any[] = [];
        const toolCallIds = new Set<string>();
        const toolResponseIds = new Set<string>();

        // First pass: collect all tool call IDs and tool response IDs
        for (const message of messages) {
            if (message.role === 'assistant' && message.tool_calls) {
                for (const toolCall of message.tool_calls) {
                    if (toolCall.id) {
                        toolCallIds.add(toolCall.id);
                    }
                }
            } else if (message.role === 'tool' && message.tool_call_id) {
                toolResponseIds.add(message.tool_call_id);
            }
        }

        // Second pass: filter out orphaned messages
        for (const message of messages) {
            if (message.role === 'assistant' && message.tool_calls) {
                // Keep assistant messages with tool calls that have corresponding responses
                const validToolCalls = message.tool_calls.filter((toolCall: any) =>
                    toolResponseIds.has(toolCall.id),
                );
                if (validToolCalls.length > 0) {
                    cleaned.push({
                        ...message,
                        tool_calls: validToolCalls,
                    });
                } else if (message.content && message.content.trim()) {
                    // Keep the message but remove tool calls if there's text content
                    cleaned.push({
                        role: message.role,
                        content: message.content,
                    });
                }
            } else if (message.role === 'tool') {
                // Keep tool messages that have corresponding tool calls
                if (toolCallIds.has(message.tool_call_id)) {
                    cleaned.push(message);
                }
            } else {
                // Keep all other messages
                cleaned.push(message);
            }
        }

        return cleaned;
    }

    private convertToGeminiFormat(completion: any): GenerateContentResponse {
        const response = new GenerateContentResponse();

        if (completion.choices && completion.choices.length > 0) {
            const choice = completion.choices[0];
            const parts: Part[] = [];

            // Handle text content
            if (choice.message?.content) {
                parts.push({ text: choice.message.content });
            }

            // Handle tool calls
            if (choice.message?.tool_calls) {
                for (const toolCall of choice.message.tool_calls) {
                    if (toolCall.type === 'function') {
                        let args: Record<string, unknown> = {};
                        try {
                            args = JSON.parse(toolCall.function.arguments || '{}');
                        } catch (error) {
                            console.warn('Failed to parse tool call arguments:', error);
                        }

                        parts.push({
                            functionCall: {
                                id: toolCall.id,
                                name: toolCall.function.name,
                                args,
                            },
                        });
                    }
                }
            }

            response.candidates = [
                {
                    content: {
                        parts,
                        role: 'model' as const,
                    },
                    finishReason: this.mapFinishReason(choice.finish_reason),
                    index: 0,
                    safetyRatings: [],
                },
            ];
        }

        // Handle usage metadata
        if (completion.usage) {
            response.usageMetadata = {
                promptTokenCount: completion.usage.prompt_tokens || 0,
                candidatesTokenCount: completion.usage.completion_tokens || 0,
                totalTokenCount: completion.usage.total_tokens || 0,
            };
        }

        response.modelVersion = this.model;
        response.promptFeedback = { safetyRatings: [] };

        return response;
    }

    private convertStreamChunkToGeminiFormat(chunk: any): GenerateContentResponse {
        const response = new GenerateContentResponse();

        if (chunk.choices && chunk.choices.length > 0) {
            const choice = chunk.choices[0];
            const parts: Part[] = [];

            // Handle text delta
            if (choice.delta?.content) {
                parts.push({ text: choice.delta.content });
            }

            // Handle tool call deltas
            if (choice.delta?.tool_calls) {
                for (const toolCallDelta of choice.delta.tool_calls) {
                    const index = toolCallDelta.index || 0;

                    if (!this.streamingToolCalls.has(index)) {
                        this.streamingToolCalls.set(index, {
                            id: undefined,
                            name: undefined,
                            arguments: '',
                        });
                    }

                    const toolCall = this.streamingToolCalls.get(index)!;

                    if (toolCallDelta.id) {
                        toolCall.id = toolCallDelta.id;
                    }
                    if (toolCallDelta.function?.name) {
                        toolCall.name = toolCallDelta.function.name;
                    }
                    if (toolCallDelta.function?.arguments) {
                        toolCall.arguments += toolCallDelta.function.arguments;
                    }

                    // If we have complete tool call info, add it to parts
                    if (toolCall.id && toolCall.name && toolCall.arguments) {
                        try {
                            const args = JSON.parse(toolCall.arguments);
                            parts.push({
                                functionCall: {
                                    id: toolCall.id,
                                    name: toolCall.name,
                                    args,
                                },
                            });
                            // Remove from streaming map since it's complete
                            this.streamingToolCalls.delete(index);
                        } catch {
                            // Arguments not complete yet, keep accumulating
                        }
                    }
                }
            }

            if (parts.length > 0) {
                response.candidates = [
                    {
                        content: {
                            parts,
                            role: 'model' as const,
                        },
                        finishReason: this.mapFinishReason(choice.finish_reason),
                        index: 0,
                        safetyRatings: [],
                    },
                ];
            }
        }

        // Handle usage metadata in final chunk
        if (chunk.usage) {
            response.usageMetadata = {
                promptTokenCount: chunk.usage.prompt_tokens || 0,
                candidatesTokenCount: chunk.usage.completion_tokens || 0,
                totalTokenCount: chunk.usage.total_tokens || 0,
            };
        }

        response.modelVersion = this.model;
        response.promptFeedback = { safetyRatings: [] };

        return response;
    }

    private mapFinishReason(reason: string | undefined): FinishReason {
        switch (reason) {
            case 'stop':
                return FinishReason.STOP;
            case 'length':
                return FinishReason.MAX_TOKENS;
            case 'tool_calls':
                return FinishReason.STOP;
            case 'content_filter':
                return FinishReason.SAFETY;
            default:
                return FinishReason.FINISH_REASON_UNSPECIFIED;
        }
    }

    private async convertGeminiRequestToTogether(
        request: GenerateContentParameters,
    ): Promise<TogetherRequestFormat> {
        const messages = this.convertToTogetherFormat(request);
        const samplingParams = this.buildSamplingParameters(request);

        const togetherRequest: TogetherRequestFormat = {
            model: this.model,
            messages,
            ...samplingParams,
        };

        if (request.config?.tools) {
            togetherRequest.tools = await this.convertGeminiToolsToTogether(
                request.config.tools,
            );
        }

        return togetherRequest;
    }

    private convertGeminiResponseToTogether(
        response: GenerateContentResponse,
    ): TogetherResponseFormat {
        const choices: TogetherChoice[] = [];

        if (response.candidates && response.candidates.length > 0) {
            const candidate = response.candidates[0];
            const message: TogetherMessage = {
                role: 'assistant',
                content: null,
            };

            const textParts: string[] = [];
            const toolCalls: TogetherToolCall[] = [];

            if (candidate.content?.parts) {
                for (const part of candidate.content.parts) {
                    if ('text' in part && part.text) {
                        textParts.push(part.text);
                    } else if ('functionCall' in part && part.functionCall) {
                        toolCalls.push({
                            id: part.functionCall.id || '',
                            type: 'function',
                            function: {
                                name: part.functionCall.name || '',
                                arguments: JSON.stringify(part.functionCall.args || {}),
                            },
                        });
                    }
                }
            }

            if (textParts.length > 0) {
                message.content = textParts.join('\n');
            }
            if (toolCalls.length > 0) {
                message.tool_calls = toolCalls;
            }

            choices.push({
                index: 0,
                message,
                finish_reason: this.mapGeminiFinishReasonToTogether(candidate.finishReason),
            });
        }

        const togetherResponse: TogetherResponseFormat = {
            id: `together-${Date.now()}`,
            object: 'chat.completion',
            created: Math.floor(Date.now() / 1000),
            model: this.model,
            choices,
        };

        if (response.usageMetadata) {
            togetherResponse.usage = {
                prompt_tokens: response.usageMetadata.promptTokenCount || 0,
                completion_tokens: response.usageMetadata.candidatesTokenCount || 0,
                total_tokens: response.usageMetadata.totalTokenCount || 0,
            };
        }

        return togetherResponse;
    }

    private mapGeminiFinishReasonToTogether(reason: FinishReason | undefined): string {
        switch (reason) {
            case FinishReason.STOP:
                return 'stop';
            case FinishReason.MAX_TOKENS:
                return 'length';
            case FinishReason.SAFETY:
                return 'content_filter';
            default:
                return 'stop';
        }
    }
}