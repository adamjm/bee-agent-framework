/**
 * Copyright 2024 IBM Corp.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { BaseAgent } from "bee-agent-framework/agents/base";
import { AnyTool } from "bee-agent-framework/tools/base";
import { BaseMemory } from "bee-agent-framework/memory/base";
import { ChatLLM, ChatLLMOutput } from "bee-agent-framework/llms/chat";
import { BaseMessage, Role } from "bee-agent-framework/llms/primitives/message";
import { AgentMeta } from "bee-agent-framework/agents/types";
import { BeeAssistantPrompt } from "bee-agent-framework/agents/bee/prompts";
import * as R from "remeda";
import { Emitter } from "bee-agent-framework/emitter/emitter";
import {
  BeeAgentRunIteration,
  BeeAgentTemplates,
  BeeCallbacks,
  BeeMeta,
  BeeRunInput,
  BeeRunOptions,
  BeeRunOutput,
} from "bee-agent-framework/agents/bee/types";
import { GetRunContext } from "bee-agent-framework/context";
import { BeeAgentRunner } from "bee-agent-framework/agents/bee/runner";
import { BeeAgentError } from "bee-agent-framework/agents/bee/errors";
import { BeeIterationToolResult } from "bee-agent-framework/agents/bee/parser";
import { assign } from "bee-agent-framework/internals/helpers/object";

import "dotenv/config.js";
import { createConsoleReader } from "../helpers/io.js";
import { FrameworkError } from "bee-agent-framework/errors";
import { TokenMemory } from "bee-agent-framework/memory/tokenMemory";
import { Logger } from "bee-agent-framework/logger/logger";
import { PythonTool } from "bee-agent-framework/tools/python/python";
import { LocalPythonStorage } from "bee-agent-framework/tools/python/storage";
import { DuckDuckGoSearchTool } from "bee-agent-framework/tools/search/duckDuckGoSearch";
import { WikipediaTool } from "bee-agent-framework/tools/search/wikipedia";
import { OpenMeteoTool } from "bee-agent-framework/tools/weather/openMeteo";
import { CalculatorTool } from "bee-agent-framework/tools/calculator";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { OllamaChatLLM } from "bee-agent-framework/adapters/ollama/chat";

/**
 * Copyright 2024 IBM Corp.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {
  BaseToolRunOptions,
  ToolError,
  ToolInputValidationError,
  ToolOutput,
} from "bee-agent-framework/tools/base";
import { getProp } from "bee-agent-framework/internals/helpers/object";
import { Retryable } from "bee-agent-framework/internals/helpers/retryable";
import { RetryCounter } from "bee-agent-framework/internals/helpers/counter";
import {
  BeeSystemPrompt,
  BeeToolErrorPrompt,
  BeeToolInputErrorPrompt,
  BeeToolNoResultsPrompt,
  BeeToolNotFoundPrompt,
  BeeUserEmptyPrompt,
  BeeUserPrompt,
} from "bee-agent-framework/agents/bee/prompts";
import { AgentError } from "bee-agent-framework/agents/base";
import { LinePrefixParser } from "bee-agent-framework/agents/parsers/linePrefix";
import { JSONParserField, ZodParserField } from "bee-agent-framework/agents/parsers/field";
import { z } from "zod";

export interface CustomRunInput {
  prompt: BaseMessage[];
}

export class BeeAgentRunnerFatalError extends BeeAgentError {
  isFatal = true;
}

export class CustomAgentRunner {
  protected readonly failedAttemptsCounter;

  constructor(
    protected readonly input: BeeInput,
    protected readonly options: BeeRunOptions,
    public readonly memory: TokenMemory,
  ) {
    this.failedAttemptsCounter = new RetryCounter(options?.execution?.totalMaxRetries, AgentError);
  }

  static async create(input: BeeInput, options: BeeRunOptions) {
    const transformMessage = (message: BaseMessage) => {
      if (message.role === Role.USER) {
        const isEmpty = !message.text.trim();
        const text = isEmpty
          ? (input.templates?.userEmpty ?? BeeUserEmptyPrompt).render({})
          : (input.templates?.user ?? BeeUserPrompt).render({
              input: message.text,
              meta: {
                ...message?.meta,
                createdAt: message?.meta?.createdAt?.toISOString?.(),
              },
            });

        return BaseMessage.of({
          role: Role.USER,
          text,
          meta: message.meta,
        });
      }
      return message;
    };

    const memory = new TokenMemory({
      llm: input.llm,
      capacityThreshold: 0.85,
      syncThreshold: 0.5,
      handlers: {
        removalSelector(curMessages) {
          // First we remove messages from the past conversations
          const prevConversationMessage = curMessages.find((msg) =>
            input.memory.messages.includes(msg),
          );
          if (prevConversationMessage) {
            return prevConversationMessage;
          }

          if (curMessages.length <= 3) {
            throw new BeeAgentRunnerFatalError(
              "Cannot fit the current conversation into the context window!",
            );
          }

          const lastMessage =
            curMessages.find(
              (msg) => msg.role === Role.ASSISTANT && getProp(msg, ["ctx", "success"]) === false,
            ) ?? curMessages.find((msg) => msg.role === Role.ASSISTANT);

          if (!lastMessage) {
            throw new BeeAgentRunnerFatalError(
              "Cannot fit the current conversation into the context window!",
            );
          }
          return lastMessage;
        },
      },
    });

    await memory.addMany([
      BaseMessage.of({
        role: Role.SYSTEM,
        text: (input.templates?.system ?? BeeSystemPrompt).render({
          tools: await Promise.all(
            input.tools.map(async (tool) => ({
              name: tool.name,
              description: tool.description.replaceAll("\n", ".").replace(/\.$/, "").concat("."),
              schema: JSON.stringify(
                await tool.getInputJsonSchema(),
                (() => {
                  const ignoredKeys = new Set(["minLength", "maxLength", "$schema"]);
                  return (key, value) => (ignoredKeys.has(key) ? undefined : value);
                })(),
              ),
            })),
          ),
          instructions: undefined,
        }),
        meta: {
          createdAt: new Date(),
        },
      }),
      ...input.memory.messages.map(transformMessage),
    ]);

    return new BeeAgentRunner(input, options, memory);
  }

  static createParser(tools: AnyTool[]) {
    const parserRegex =
      /Thought:.+\n(?:Final Answer:[\S\s]+|Function Name:.+\nFunction Input: \{.*\}\nFunction Caption:.+\nFunction Output:)?/;

    const parser = new LinePrefixParser(
      {
        thought: {
          prefix: "Thought:",
          next: ["tool_name", "final_answer"],
          isStart: true,
          field: new ZodParserField(z.string().min(1)),
        },
        tool_name: {
          prefix: "Function Name:",
          next: ["tool_input"],
          field: new ZodParserField(
            z.pipeline(
              z.string().trim(),
              z.enum(tools.map((tool) => tool.name) as [string, ...string[]]),
            ),
          ),
        },
        tool_input: {
          prefix: "Function Input:",
          next: ["tool_caption", "tool_output"],
          isEnd: true,
          field: new JSONParserField({
            schema: z.object({}).passthrough(),
            base: {},
            matchPair: ["{", "}"],
          }),
        },
        tool_caption: {
          prefix: "Function Caption:",
          next: ["tool_output"],
          isEnd: true,
          field: new ZodParserField(z.string()),
        },
        tool_output: {
          prefix: "Function Output:",
          next: ["final_answer"],
          isEnd: true,
          field: new ZodParserField(z.string()),
        },
        final_answer: {
          prefix: "Final Answer:",
          next: [],
          isStart: true,
          isEnd: true,
          field: new ZodParserField(z.string().min(1)),
        },
      } as const,
      {
        waitForStartNode: true,
        endOnRepeat: true,
        fallback: (stash) =>
          stash
            ? [
                { key: "thought", value: "I now know the final answer." },
                { key: "final_answer", value: stash },
              ]
            : [],
      },
    );

    return {
      parser,
      parserRegex,
    } as const;
  }

  async llm(input: {
    emitter: Emitter<BeeCallbacks>;
    signal: AbortSignal;
    meta: BeeMeta;
  }): Promise<BeeAgentRunIteration> {
    const { emitter, signal, meta } = input;

    return new Retryable({
      onRetry: () => emitter.emit("retry", { meta }),
      onError: async (error) => {
        await emitter.emit("error", { error, meta });
        this.failedAttemptsCounter.use(error);
      },
      executor: async () => {
        await emitter.emit("start", { meta });

        const { parser, parserRegex } = BeeAgentRunner.createParser(this.input.tools);
        const llmOutput = await this.input.llm
          .generate(this.memory.messages.slice(), {
            signal,
            stream: true,
            guided: {
              regex: parserRegex.source,
            },
          })
          .observe((llmEmitter) => {
            parser.emitter.on("update", async ({ value, key, field }) => {
              if (key === "tool_output" && parser.isDone) {
                return;
              }
              await emitter.emit("update", {
                data: parser.finalState,
                update: { key, value: field.raw, parsedValue: value },
                meta: { success: true, ...meta },
              });
            });
            parser.emitter.on("partialUpdate", async ({ key, delta, value }) => {
              await emitter.emit("partialUpdate", {
                data: parser.finalState,
                update: { key, value: delta, parsedValue: value },
                meta: { success: true, ...meta },
              });
            });

            llmEmitter.on("newToken", async ({ value, callbacks }) => {
              if (parser.isDone) {
                callbacks.abort();
                return;
              }

              await parser.add(value.getTextContent());
              if (parser.partialState.tool_output !== undefined) {
                callbacks.abort();
              }
            });
          });

        await parser.end();

        return {
          state: parser.finalState,
          raw: llmOutput,
        };
      },
      config: {
        maxRetries: this.options.execution?.maxRetriesPerStep,
        signal,
      },
    }).get();
  }

  async tool(input: {
    iteration: BeeIterationToolResult;
    signal: AbortSignal;
    emitter: Emitter<BeeCallbacks>;
    meta: BeeMeta;
  }): Promise<{ output: string; success: boolean }> {
    const { iteration, signal, emitter, meta } = input;

    const tool = this.input.tools.find(
      (tool) => tool.name.trim().toUpperCase() == iteration.tool_name?.toUpperCase(),
    );
    if (!tool) {
      this.failedAttemptsCounter.use(
        new AgentError(`Agent was trying to use non-existing tool "${iteration.tool_name}"`, [], {
          context: { iteration, meta },
        }),
      );

      const template = this.input.templates?.toolNotFoundError ?? BeeToolNotFoundPrompt;
      return {
        success: false,
        output: template.render({
          tools: this.input.tools,
        }),
      };
    }
    const options = await (async () => {
      const baseOptions: BaseToolRunOptions = {
        signal,
      };
      const customOptions = await this.options.modifiers?.getToolRunOptions?.({
        tool,
        input: iteration.tool_input,
        baseOptions,
      });
      return customOptions ?? baseOptions;
    })();

    return new Retryable({
      config: {
        signal,
        maxRetries: this.options.execution?.maxRetriesPerStep,
      },
      onError: async (error) => {
        await emitter.emit("toolError", {
          data: {
            iteration,
            tool,
            input: iteration.tool_input,
            options,
            error: FrameworkError.ensure(error),
          },
          meta,
        });
        this.failedAttemptsCounter.use(error);
      },
      executor: async () => {
        try {
          await emitter.emit("toolStart", {
            data: {
              tool,
              input: iteration.tool_input,
              options,
              iteration,
            },
            meta,
          });
          const toolOutput: ToolOutput = await tool.run(iteration.tool_input, options);
          await emitter.emit("toolSuccess", {
            data: {
              tool,
              input: iteration.tool_input,
              options,
              result: toolOutput,
              iteration,
            },
            meta,
          });

          if (toolOutput.isEmpty()) {
            const template = this.input.templates?.toolNoResultError ?? BeeToolNoResultsPrompt;
            return { output: template.render({}), success: true };
          }

          return {
            success: true,
            output: toolOutput.getTextContent(),
          };
        } catch (error) {
          await emitter.emit("toolError", {
            data: {
              tool,
              input: iteration.tool_input,
              options,
              error,
              iteration,
            },
            meta,
          });

          if (error instanceof ToolInputValidationError) {
            this.failedAttemptsCounter.use(error);

            const template = this.input.templates?.toolInputError ?? BeeToolInputErrorPrompt;
            return {
              success: false,
              output: template.render({
                reason: error.toString(),
              }),
            };
          }

          if (error instanceof ToolError) {
            this.failedAttemptsCounter.use(error);

            const template = this.input.templates?.toolError ?? BeeToolErrorPrompt;
            return {
              success: false,
              output: template.render({
                reason: error.explain(),
              }),
            };
          }

          throw error;
        }
      },
    }).get();
  }
}

export interface BeeInput {
  llm: ChatLLM<ChatLLMOutput>;
  tools: AnyTool[];
  memory: BaseMemory;
  meta?: Omit<AgentMeta, "tools">;
  templates?: Partial<BeeAgentTemplates>;
}

export class CustomAgent extends BaseAgent<CustomRunInput, BeeRunOutput, BeeRunOptions> {
  public readonly emitter = Emitter.root.child<BeeCallbacks>({
    namespace: ["agent", "bee"],
    creator: this,
  });

  constructor(protected readonly input: BeeInput) {
    super();

    const duplicate = input.tools.find((a, i, arr) =>
      arr.find((b, j) => i !== j && a.name.toUpperCase() === b.name.toUpperCase()),
    );
    if (duplicate) {
      throw new BeeAgentError(
        `Agent's tools must all have different names. Conflicting tool: ${duplicate.name}.`,
      );
    }
  }

  static {
    this.register();
  }

  get memory() {
    return this.input.memory;
  }

  get meta(): AgentMeta {
    const tools = this.input.tools.slice();

    if (this.input.meta) {
      return { ...this.input.meta, tools };
    }

    return {
      name: "Bee",
      tools,
      description:
        "The Bee framework demonstrates its ability to auto-correct and adapt in real-time, improving the overall reliability and resilience of the system.",
      ...(tools.length > 0 && {
        extraDescription: [
          `Tools that I can use to accomplish given task.`,
          ...tools.map((tool) => `Tool '${tool.name}': ${tool.description}.`),
        ].join("\n"),
      }),
    };
  }

  protected async _run(
    input: CustomRunInput,
    options: BeeRunOptions = {},
    run: GetRunContext<typeof this>,
  ): Promise<BeeRunOutput> {
    // Run initial prompt -> gets converted to base message (in memory)
    // Get tool call information
    // IF TOOL CALL
    // Stop
    //
    //
    // IF TOOL OUTPUT -> add to memory so that add to prompt
    // Add tool output to prompt
    // Run prompt
    // IF FINAL OUTPUT
    // Get tool call information OR get final output
    //
    //
    // Need the runner to take in a BaseMessage or series of base messages as memory

    const iterations: BeeAgentRunIteration[] = [];

    const runner = await CustomAgentRunner.create(this.input, options);
    for (const message of input.prompt) {
      await runner.memory.add(message);
    }
    let finalMessage: BaseMessage | undefined;
    let resp: BaseMessage;
    const meta: BeeMeta = { iteration: iterations.length + 1 };

    const emitter = run.emitter.child({ groupId: `iteration-${meta.iteration}` });
    const iteration = await runner.llm({ emitter, signal: run.signal, meta });

    // tool run here.
    if (iteration.state.tool_name || iteration.state.tool_caption || iteration.state.tool_input) {
      // return tool call here
      //
      //
      const output = "test";
      const success = true;
      console.log(
        iteration.state.tool_name,
        iteration.state.tool_caption,
        iteration.state.tool_input,
        iteration.state.final_answer,
      );
      resp = BaseMessage.of({
        role: Role.ASSISTANT,
        text: BeeAssistantPrompt.clone().render({
          toolName: [iteration.state.tool_name].filter(R.isTruthy),
          toolCaption: [iteration.state.tool_caption].filter(R.isTruthy),
          toolInput: [iteration.state.tool_input]
            .filter(R.isTruthy)
            .map((call) => JSON.stringify(call)),
          thought: [iteration.state.thought].filter(R.isTruthy),
          finalAnswer: [iteration.state.final_answer].filter(R.isTruthy),
          toolOutput: [output],
        }),
        meta: { success },
      });

      return { result: resp, iterations, memory: runner.memory };

      // const { output, success } = await runner.tool({
      //   iteration: iteration.state as BeeIterationToolResult,
      //   signal: run.signal,
      //   emitter,
      //   meta,
      // });

      // for (const key of ["partialUpdate", "update"] as const) {
      //   await emitter.emit(key, {
      //     data: {
      //       ...iteration.state,
      //       tool_output: output,
      //     },
      //     update: { key: "tool_output", value: output, parsedValue: output },
      //     meta: { success, ...meta },
      //   });
      // }

      // resp = BaseMessage.of({
      //   role: Role.ASSISTANT,
      //   text: BeeAssistantPrompt.clone().render({
      //     toolName: [iteration.state.tool_name].filter(R.isTruthy),
      //     toolCaption: [iteration.state.tool_caption].filter(R.isTruthy),
      //     toolInput: [iteration.state.tool_input]
      //       .filter(R.isTruthy)
      //       .map((call) => JSON.stringify(call)),
      //     thought: [iteration.state.thought].filter(R.isTruthy),
      //     finalAnswer: [iteration.state.final_answer].filter(R.isTruthy),
      //     toolOutput: [output],
      //   }),
      //   meta: { success },
      // });

      // // memory can be handled outside of the agent call
      // await runner.memory.add(resp);

      // assign(iteration.state, { tool_output: output });
    }
    if (iteration.state.final_answer) {
      resp = BaseMessage.of({
        role: Role.ASSISTANT,
        text: iteration.state.final_answer,
        meta: {
          createdAt: new Date(),
        },
      });
      await run.emitter.emit("success", {
        data: finalMessage,
        iterations,
        memory: runner.memory,
        meta,
      });
      await runner.memory.add(resp);

      return { result: resp, iterations, memory: runner.memory };
    }
    resp = BaseMessage.of({
      role: Role.ASSISTANT,
      text: "oh I don't think I quite got that right",
    });

    return { result: resp, iterations, memory: runner.memory };
  }

  createSnapshot() {
    return {
      input: this.input,
      emitter: this.emitter,
    };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}

Logger.root.level = "silent"; // disable internal logs
const logger = new Logger({ name: "app", level: "trace" });

const llm = new OllamaChatLLM({
  modelId: "llama3.1:8b", // "granite3-dense:8b", // llama3.1:70b for better performance
});

const codeInterpreterUrl = process.env.CODE_INTERPRETER_URL;
const __dirname = dirname(fileURLToPath(import.meta.url));

const codeInterpreterTmpdir =
  process.env.CODE_INTEPRETER_TMPDIR ?? "./examples/tmp/code_interpreter";
const localTmpdir = process.env.LOCAL_TMPDIR ?? "./examples/tmp/local";

const agent = new CustomAgent({
  llm,
  memory: new TokenMemory({ llm }),
  tools: [
    new DuckDuckGoSearchTool(),
    // new WebCrawlerTool(), // HTML web page crawler
    new WikipediaTool(),
    new OpenMeteoTool(), // weather tool
    new CalculatorTool(), //calculator
    // new ArXivTool(), // research papers
    // new DynamicTool() // custom python tool
    ...(codeInterpreterUrl
      ? [
          new PythonTool({
            codeInterpreter: { url: codeInterpreterUrl },
            storage: new LocalPythonStorage({
              interpreterWorkingDir: `${__dirname}/../../${codeInterpreterTmpdir}`,
              localWorkingDir: `${__dirname}/../../${localTmpdir}`,
            }),
          }),
        ]
      : []),
  ],
});
const user_input = "Do the following calculation 1+1";
const user_in = BaseMessage.of({
  role: Role.USER,
  text: user_input ?? "",
  meta: {
    // TODO: createdAt
    createdAt: new Date(),
  },
});

const tool = BaseMessage.of({
  role: Role.ASSISTANT,
  text: BeeAssistantPrompt.clone().render({
    toolName: ["Calculator"].filter(R.isTruthy),
    toolCaption: ["Perform the calculation 1+1."].filter(R.isTruthy),
    toolInput: ["{ expression: '1+1' }"].filter(R.isTruthy).map((call) => JSON.stringify(call)),
    thought: [
      "The user wants to perform a calculation. I can use the Calculator function to do that.",
    ].filter(R.isTruthy),
    finalAnswer: [undefined].filter(R.isTruthy),
    toolOutput: ["2"],
  }),
});

const prompt = [user_in, tool];

const response = await agent
  .run(
    { prompt },
    {
      execution: {
        maxRetriesPerStep: 3,
        totalMaxRetries: 10,
        maxIterations: 20,
      },
    },
  )
  .observe((emitter) => {
    // emitter.on("start", () => {
    //   reader.write(`Agent 🤖 : `, "starting new iteration");
    // });
    emitter.on("error", ({ error }) => {
      console.info(`Agent 🤖 : `, FrameworkError.ensure(error).dump());
    });
    emitter.on("retry", () => {
      console.info(`Agent 🤖 : `, "retrying the action...");
    });
    emitter.on("update", async ({ data, update, meta }) => {
      // log 'data' to see the whole state
      // to log only valid runs (no errors), check if meta.success === true
      console.info(`Agent (${update.key}) 🤖 : `, update.value);
    });
    emitter.on("partialUpdate", ({ data, update, meta }) => {
      // ideal for streaming (line by line)
      // log 'data' to see the whole state
      // to log only valid runs (no errors), check if meta.success === true
      // console.info(`Agent (partial ${update.key}) 🤖 : `, update.value);
    });

    // To observe all events (uncomment following block)
    // emitter.match("*.*", async (data: unknown, event) => {
    //   logger.trace(event, `Received event "${event.path}"`);
    // });

    // To get raw LLM input (uncomment following block)
    // emitter.match(
    //   (event) => event.creator === llm && event.name === "start",
    //   async (data: InferCallbackValue<GenerateCallbacks["start"]>, event) => {
    //     logger.trace(
    //       event,
    //       [
    //         `Received LLM event "${event.path}"`,
    //         JSON.stringify(data.input), // array of messages
    //       ].join("\n"),
    //     );
    //   },
    // );
  });

console.info(`Agent 🤖 : `, response.result.text);
