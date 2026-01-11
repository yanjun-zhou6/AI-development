import { OpenRouter } from "@openrouter/sdk";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import readline from "readline/promises";
import dotenv from "dotenv";
import { Message, ToolDefinitionJson } from "@openrouter/sdk/models";

dotenv.config();

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
if (!OPENROUTER_API_KEY) {
  throw new Error("OPENROUTER_API_KEY is not set");
}

class McpClient {
  private mcp: Client;
  private openRouter: OpenRouter;
  private transport: StdioClientTransport | null = null;
  private tools: ToolDefinitionJson[] = [];

  constructor() {
    this.openRouter = new OpenRouter({
      apiKey: OPENROUTER_API_KEY,
    });

    this.mcp = new Client({ name: "mcp-client-cli", version: "1.0.0" });
  }

  async connectToServer(serverScriptPath: string) {
    try {
      const isJs = serverScriptPath.endsWith(".js");
      const isPy = serverScriptPath.endsWith(".py");

      if (!isJs && !isPy) {
        throw new Error("Server script must be a .js or .py file");
      }

      const command = isPy
        ? process.platform === "win32"
          ? "python"
          : "python3"
        : process.execPath;

      this.transport = new StdioClientTransport({
        command,
        args: [serverScriptPath],
      });

      await this.mcp.connect(this.transport);
      const toolsResult = await this.mcp.listTools();

      this.tools = toolsResult.tools.map((tool) => {
        return {
          type: "function",
          function: {
            name: tool.name,
            description: tool.description,
            parameters: tool.inputSchema,
          },
        };
      });

      console.log(
        "Connected to server with tools:",
        this.tools.map((tool) => tool.function.name)
      );
    } catch (e) {
      console.log("Failed to connect to MCP server: ", e);
      throw e;
    }
  }

  async processQuery(query: string) {
    // Create a system prompt that describes available tools
    const toolDescriptions = this.tools
      .map(
        (tool) =>
          `- ${tool.function.name}: ${
            tool.function.description
          }\n  Parameters: ${JSON.stringify(tool.function.parameters, null, 2)}`
      )
      .join("\n");

    const systemPrompt = `You are an AI assistant with access to the following tools:

${toolDescriptions}

To use a tool, respond with a JSON object in this exact format:
{
  "tool_call": {
    "name": "tool_name",
    "arguments": {"param1": "value1", "param2": "value2"}
  }
}

If you don't need to use any tools, just respond normally. Only use tools when specifically needed to answer the user's question.`;

    const messages: Message[] = [
      {
        role: "system",
        content: systemPrompt,
      },
      {
        role: "user",
        content: query,
      },
    ];

    const response = await this.openRouter.chat.send({
      model: "meta-llama/llama-3.2-3b-instruct:free",
      messages,
      maxTokens: 1000,
      stream: false,
    });

    const finalText = [];
    const message = response.choices[0].message;
    let content = "";

    // Handle different content types
    if (typeof message.content === "string") {
      content = message.content;
    } else if (Array.isArray(message.content)) {
      // Extract text from content array
      content = message.content
        .filter((item: any) => item.type === "text")
        .map((item: any) => item.text)
        .join("");
    }

    // Check if the response contains a tool call - improved regex to handle nested objects
    const toolCallMatch = content.match(
      /\{\s*"tool_call"\s*:\s*\{[^{}]*\{[^}]*\}[^}]*\}\s*\}/
    );

    if (toolCallMatch) {
      try {
        // Extract just the JSON part and clean it
        let jsonString = toolCallMatch[0];

        // Try to find a more complete JSON if the regex didn't capture it fully
        const startIndex = content.indexOf('{"tool_call"');
        if (startIndex !== -1) {
          let braceCount = 0;
          let endIndex = startIndex;

          for (let i = startIndex; i < content.length; i++) {
            if (content[i] === "{") braceCount++;
            if (content[i] === "}") braceCount--;
            if (braceCount === 0) {
              endIndex = i;
              break;
            }
          }

          if (endIndex > startIndex) {
            jsonString = content.substring(startIndex, endIndex + 1);
          }
        }

        console.log("Attempting to parse JSON:", jsonString);
        const toolCallData = JSON.parse(jsonString);
        const toolName = toolCallData.tool_call.name;
        const toolArgs = toolCallData.tool_call.arguments;

        // Convert string arguments to proper types if needed
        const processedArgs: any = {};
        for (const [key, value] of Object.entries(toolArgs)) {
          if (typeof value === "string" && !isNaN(Number(value))) {
            processedArgs[key] = Number(value);
          } else {
            processedArgs[key] = value;
          }
        }

        // Execute the tool
        const result = await this.mcp.callTool({
          name: toolName,
          arguments: processedArgs,
        });

        finalText.push(
          `[Calling tool ${toolName}] with args ${JSON.stringify(
            processedArgs
          )}`
        );

        // Get follow-up response with tool result
        messages.push({
          role: "assistant",
          content: content,
        });

        messages.push({
          role: "user",
          content: `Tool result: ${JSON.stringify(
            result.content
          )}. Please provide a final response based on this information.`,
        });

        const followUpResponse = await this.openRouter.chat.send({
          model: "meta-llama/llama-3.2-3b-instruct:free",
          messages,
          maxTokens: 1000,
          stream: false,
        });

        const followUpMessage = followUpResponse.choices[0].message;
        let followUpContent = "";

        if (typeof followUpMessage.content === "string") {
          followUpContent = followUpMessage.content;
        } else if (Array.isArray(followUpMessage.content)) {
          followUpContent = followUpMessage.content
            .filter((item: any) => item.type === "text")
            .map((item: any) => item.text)
            .join("");
        }

        if (followUpContent) {
          finalText.push(followUpContent);
        }
      } catch (error) {
        console.error("Full content:", content);
        console.error("Matched JSON:", toolCallMatch[0]);
        finalText.push("Error parsing tool call: " + error);
        finalText.push(content);
      }
    } else {
      // No tool call, just return the response
      finalText.push(content);
    }

    return finalText.join("\n");
  }

  async chatLoop() {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    try {
      console.log("\nMCP Client Started!");
      console.log("Type your queries or 'quit' to exit.");

      while (true) {
        const message = await rl.question("\nQuery: ");
        if (message.toLowerCase() === "quit") {
          break;
        }
        const response = await this.processQuery(message);
        console.log("\n" + response);
      }
    } finally {
      rl.close();
    }
  }

  async cleanup() {
    await this.mcp.close();
  }
}

async function main() {
  if (process.argv.length < 3) {
    console.log("Usage: node index.ts <path_to_server_script>");
    return;
  }
  const mcpClient = new McpClient();
  try {
    await mcpClient.connectToServer(process.argv[2]);
    await mcpClient.chatLoop();
  } catch (e) {
    console.error("Error:", e);
    await mcpClient.cleanup();
    process.exit(1);
  } finally {
    await mcpClient.cleanup();
    process.exit(0);
  }
}

main();
