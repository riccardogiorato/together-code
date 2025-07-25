/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

type Model = string;
type TokenCount = number;

export const DEFAULT_TOKEN_LIMIT = 32_000; //32K context window

// TODO add most of the together AI models that work correctly with CLI

export function tokenLimit(model: Model): TokenCount {
  // Add other models as they become relevant or if specified by config
  // Find more models here https://www.together.ai/models
  switch (model) {
    case "moonshotai/Kimi-K2-Instruct":
      return 128_000;
    case "Qwen/Qwen3-235B-A22B-Instruct-2507-tput":
    case "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8":
      return 256_000;
    default:
      return DEFAULT_TOKEN_LIMIT;
  }
}
