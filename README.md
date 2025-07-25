# Together Code

![Together Code Screenshot](./docs/assets/qwen-screenshot.png)

**Together Code** is a command-line AI workflow tool forked from [**Qwen Code**](https://github.com/QwenLM/qwen-code), which itself was forked from [**Gemini CLI**](https://github.com/google-gemini/gemini-cli) (Please refer to [this document](./README.gemini.md) for more details). This project continues the lineage of AI-powered development tools, building upon the foundations established by Gemini CLI and enhanced by Qwen Code.

> [!WARNING]
> Qwen Code may issue multiple API calls per cycle, resulting in higher token usage, similar to Claude Code. We’re actively working to enhance API efficiency and improve the overall developer experience.

## Key Features

- **Code Understanding & Editing** - Query and edit large codebases beyond traditional context window limits
- **Workflow Automation** - Automate operational tasks like handling pull requests and complex rebases
- **Enhanced Parser** - Adapted parser specifically optimized for Qwen-Coder models

## Quick Start

### Prerequisites

Ensure you have [Node.js version 20](https://nodejs.org/en/download) or higher installed.

```bash
curl -qL https://www.npmjs.com/install.sh | sh
```

### Installation

```bash
npm install -g @qwen-code/qwen-code
qwen --version
```

Then run from anywhere:

```bash
qwen
```

Or you can install it from source:

```bash
git clone https://github.com/QwenLM/qwen-code.git
cd qwen-code
npm install
npm install -g .
```

### API Configuration

Set your Together AI API key. You can obtain your API key from [https://togetherai.link](https://togetherai.link).

```bash
export TOGETHER_API_KEY="your_api_key_here"
export TOGETHER_MODEL="your_model_name_here"
```

## Usage Examples

### Explore Codebases

```sh
cd your-project/
qwen
> Describe the main pieces of this system's architecture
```

### Code Development

```sh
> Refactor this function to improve readability and performance
```

### Automate Workflows

```sh
> Analyze git commits from the last 7 days, grouped by feature and team member
```

```sh
> Convert all images in this directory to PNG format
```

## Popular Tasks

### Understand New Codebases

```text
> What are the core business logic components?
> What security mechanisms are in place?
> How does the data flow work?
```

### Code Refactoring & Optimization

```text
> What parts of this module can be optimized?
> Help me refactor this class to follow better design patterns
> Add proper error handling and logging
```

### Documentation & Testing

```text
> Generate comprehensive JSDoc comments for this function
> Write unit tests for this component
> Create API documentation
```

## Benchmark Results

### Terminal-Bench

| Agent     | Model              | Accuracy |
| --------- | ------------------ | -------- |
| Qwen Code | Qwen3-Coder-480A35 | 37.5     |

## Project Structure

```
qwen-code/
├── packages/           # Core packages
├── docs/              # Documentation
├── examples/          # Example code
└── tests/            # Test files
```

## Development & Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) to learn how to contribute to the project.

## Troubleshooting

If you encounter issues, check the [troubleshooting guide](docs/troubleshooting.md).

## Acknowledgments

This project is a fork of [Qwen Code](https://github.com/QwenLM/qwen-code), which itself is a fork of [Google Gemini CLI](https://github.com/google-gemini/gemini-cli). We acknowledge and appreciate the excellent work of both the Gemini CLI and Qwen Code teams. Our contributions build upon their foundations.

## License

[LICENSE](./LICENSE)

<!-- ## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/qwen-code&type=Date)](https://www.star-history.com/#QwenLM/qwen-code&Date) -->
